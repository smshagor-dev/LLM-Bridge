//! Authenticate a complete checkpoint, then verify exact blocks before consumption.
use super::BoundedLocalReader;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub struct SealedMetadata {
    _file: File,
    path: PathBuf,
    pub prefix_bytes: u64,
}
impl SealedMetadata {
    pub fn path(&self) -> &Path {
        &self.path
    }
}
#[derive(Debug, Clone, Default)]
pub struct AuthenticationMetrics {
    pub expected_sha256: String,
    pub accounted_bytes: usize,
    pub blocks_verified: u64,
}
pub(super) struct Authentication {
    hashes: Vec<[u8; 32]>,
    scratch: Vec<u8>,
    cached_block: Option<usize>,
    metrics: AuthenticationMetrics,
}
fn fail(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl BoundedLocalReader {
    pub fn authentication_metrics(&self) -> Option<AuthenticationMetrics> {
        self.auth.as_ref().map(|a| a.metrics.clone())
    }
    /// Whole-file SHA-256 anchors a bounded block manifest. Only the verified
    /// metadata prefix is retained in a sealed Linux memfd; weight pages are holes.
    /// The returned handle must outlive native vocabulary loading.
    #[cfg(target_os = "linux")]
    pub fn authenticate_model(
        &mut self,
        expected: &str,
        prefix_bytes: u64,
        overhead_budget: usize,
        mut cancelled: impl FnMut() -> bool,
    ) -> io::Result<SealedMetadata> {
        use std::io::Write;
        use std::os::fd::{AsRawFd, FromRawFd};
        if self.auth.is_some()
            || expected.len() != 64
            || !expected.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(fail(
                "invalid expected SHA-256 or reader already authenticated",
            ));
        }
        if prefix_bytes == 0 || prefix_bytes > self.file_len {
            return Err(fail("invalid tokenizer metadata prefix"));
        }
        let chunk = self.buffer.len();
        let blocks = self.file_len.div_ceil(chunk as u64);
        let hash_bytes = usize::try_from(blocks)
            .ok()
            .and_then(|n| n.checked_mul(32))
            .ok_or_else(|| fail("authentication manifest overflow"))?;
        // Account page-rounded memfd metadata, hash table, one read scratch and digest string.
        let page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| fail("OS page size unavailable"))?;
        let metadata_bytes = usize::try_from(prefix_bytes)
            .ok()
            .and_then(|n| n.checked_add(page_size - 1))
            .map(|n| n / page_size * page_size)
            .ok_or_else(|| fail("metadata accounting overflow"))?;
        let accounted = hash_bytes
            .checked_add(metadata_bytes)
            .and_then(|n| n.checked_add(chunk))
            .and_then(|n| n.checked_add(64))
            .ok_or_else(|| fail("authentication accounting overflow"))?;
        if accounted > overhead_budget {
            return Err(fail(
                "authenticated metadata/manifest exceeds overhead budget",
            ));
        }
        if cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "authentication cancelled",
            ));
        }
        let mut hashes = Vec::new();
        hashes.try_reserve_exact(blocks as usize).map_err(|_| {
            io::Error::new(io::ErrorKind::OutOfMemory, "manifest allocation failed")
        })?;
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(chunk).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "authentication scratch allocation failed",
            )
        })?;
        scratch.resize(chunk, 0);
        // SAFETY: fixed NUL-terminated name; owned descriptor is closed by File.
        let fd = unsafe {
            libc::memfd_create(
                c"llmbridge-vocabulary".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut metadata = unsafe { File::from_raw_fd(fd) };
        metadata.set_len(self.file_len)?;
        let mut whole = Sha256::new();
        let mut prefix_hash = Sha256::new();
        let mut offset = 0;
        while offset < self.file_len {
            let count = (self.file_len - offset).min(chunk as u64) as usize;
            let bytes = self.read_range_cancellable(offset, count, &mut cancelled)?;
            whole.update(bytes);
            hashes.push(Sha256::digest(bytes).into());
            if offset < prefix_bytes {
                let keep = (prefix_bytes - offset).min(count as u64) as usize;
                prefix_hash.update(&bytes[..keep]);
                metadata.write_all(&bytes[..keep])?;
            }
            offset += count as u64;
        }
        if cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "authentication cancelled",
            ));
        }
        self.validate_source()?;
        let actual = format!("{:x}", whole.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(fail("SHA-256 mismatch; native execution not admitted"));
        }
        let seals =
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        if unsafe { libc::fcntl(metadata.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
            return Err(io::Error::last_os_error());
        }
        let sealed = unsafe { libc::fcntl(metadata.as_raw_fd(), libc::F_GET_SEALS) };
        if sealed < 0 || sealed & seals != seals {
            return Err(fail("tokenizer metadata sealing failed"));
        }
        // Rehash after sealing: even a write through another descriptor during
        // construction must not freeze attacker-modified tokenizer metadata.
        metadata.seek(SeekFrom::Start(0))?;
        let mut sealed_hash = Sha256::new();
        let mut remaining = prefix_bytes;
        while remaining > 0 {
            if cancelled() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "authentication cancelled",
                ));
            }
            let count = remaining.min(chunk as u64) as usize;
            metadata.read_exact(&mut scratch[..count])?;
            sealed_hash.update(&scratch[..count]);
            remaining -= count as u64;
        }
        if sealed_hash.finalize() != prefix_hash.finalize() {
            return Err(fail("sealed tokenizer metadata digest mismatch"));
        }
        let path = PathBuf::from(format!("/proc/self/fd/{}", metadata.as_raw_fd()));
        self.auth = Some(Authentication {
            hashes,
            scratch,
            cached_block: None,
            metrics: AuthenticationMetrics {
                expected_sha256: actual,
                accounted_bytes: accounted,
                blocks_verified: 0,
            },
        });
        Ok(SealedMetadata {
            _file: metadata,
            path,
            prefix_bytes,
        })
    }
    #[cfg(not(target_os = "linux"))]
    pub fn authenticate_model(
        &mut self,
        _expected: &str,
        _prefix_bytes: u64,
        _budget: usize,
        _cancelled: impl FnMut() -> bool,
    ) -> io::Result<SealedMetadata> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "sealed tokenizer authentication currently requires Linux",
        ))
    }
    pub(super) fn read_authenticated_range(
        &mut self,
        offset: u64,
        count: usize,
    ) -> io::Result<&[u8]> {
        self.validate_source()?;
        let auth = self.auth.as_mut().expect("authenticated reader");
        // A previous read may have verified this block before an in-place rewrite.
        // Rehash on each public range request, even when file identity is unchanged.
        auth.cached_block = None;
        let chunk = self.buffer.len();
        let mut copied = 0;
        while copied < count {
            let current = offset + copied as u64;
            let block = usize::try_from(current / chunk as u64)
                .map_err(|_| fail("manifest block overflow"))?;
            let start = (block as u64) * chunk as u64;
            let available = (self.file_len - start).min(chunk as u64) as usize;
            if auth.cached_block != Some(block) {
                auth.cached_block = None;
                let next_bytes = self
                    .disk_bytes_read
                    .checked_add(available as u64)
                    .ok_or_else(|| fail("authenticated I/O counter overflow"))?;
                self.file.seek(SeekFrom::Start(start))?;
                self.file.read_exact(&mut auth.scratch[..available])?;
                self.disk_bytes_read = next_bytes;
                let digest: [u8; 32] = Sha256::digest(&auth.scratch[..available]).into();
                if auth.hashes.get(block) != Some(&digest) {
                    return Err(fail(
                        "authenticated model block changed; reject tensor bytes",
                    ));
                }
                auth.metrics.blocks_verified = auth
                    .metrics
                    .blocks_verified
                    .checked_add(1)
                    .ok_or_else(|| fail("verified block counter overflow"))?;
                auth.cached_block = Some(block);
            }
            let within = (current - start) as usize;
            let bytes = (count - copied).min(available - within);
            self.buffer[copied..copied + bytes]
                .copy_from_slice(&auth.scratch[within..within + bytes]);
            copied += bytes;
        }
        self.validate_source()?;
        Ok(&self.buffer[..count])
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "bridge-auth-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&p, b"abcdefghijklmnopqrstuvwxyz").unwrap();
            Self(p)
        }
        fn reader(&self) -> BoundedLocalReader {
            BoundedLocalReader::open(&self.0, 8, 16).unwrap()
        }
        fn digest(&self) -> String {
            format!("{:x}", Sha256::digest(std::fs::read(&self.0).unwrap()))
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    #[test]
    fn sealed_metadata_has_only_verified_prefix_and_cannot_be_rewritten() {
        let file = Fixture::new();
        let mut reader = file.reader();
        let snapshot = reader
            .authenticate_model(&file.digest(), 10, 128 * 1024, || false)
            .unwrap();
        let mut bytes = Vec::new();
        File::open(snapshot.path())
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(&bytes[..10], b"abcdefghij");
        assert!(bytes[10..].iter().all(|x| *x == 0));
        assert!(std::fs::OpenOptions::new()
            .write(true)
            .open(snapshot.path())
            .unwrap()
            .write_all(b"X")
            .is_err());
        assert_eq!(reader.read_range(6, 6).unwrap(), b"ghijkl");
        assert_eq!(reader.read_range(24, 2).unwrap(), b"yz");
        assert_eq!(reader.authentication_metrics().unwrap().blocks_verified, 3);
    }
    #[test]
    fn bad_digest_and_bad_budget_never_enable_authenticated_reads() {
        let file = Fixture::new();
        let mut reader = file.reader();
        assert!(reader
            .authenticate_model("bad", 10, 128 * 1024, || false)
            .is_err());
        assert_eq!(reader.disk_bytes_read(), 0);
        assert!(reader
            .authenticate_model(&file.digest(), 10, 1, || false)
            .is_err());
        assert_eq!(reader.disk_bytes_read(), 0);
        assert!(reader
            .authenticate_model(&"0".repeat(64), 10, 128 * 1024, || false)
            .is_err());
        assert!(reader.authentication_metrics().is_none());
    }
    #[test]
    fn unchanged_identity_cannot_hide_a_modified_uncached_block() {
        let file = Fixture::new();
        let mut reader = file.reader();
        let _sealed = reader
            .authenticate_model(&file.digest(), 10, 128 * 1024, || false)
            .unwrap();
        let modified = std::fs::metadata(&file.0).unwrap().modified().unwrap();
        reader.read_range(0, 2).unwrap();
        let mut writable = std::fs::OpenOptions::new()
            .write(true)
            .open(&file.0)
            .unwrap();
        writable.seek(SeekFrom::Start(16)).unwrap();
        writable.write_all(b"X").unwrap();
        writable
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert!(
            reader.validate_source().is_ok(),
            "identity deliberately unchanged"
        );
        assert!(reader
            .read_range(16, 2)
            .unwrap_err()
            .to_string()
            .contains("block changed"));
    }
    #[test]
    fn cancelled_authentication_never_publishes_a_session() {
        let file = Fixture::new();
        let mut reader = file.reader();
        let mut calls = 0;
        let error = reader
            .authenticate_model(&file.digest(), 10, 128 * 1024, || {
                calls += 1;
                calls == 3
            })
            .err()
            .unwrap();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert!(reader.authentication_metrics().is_none());
    }
}
