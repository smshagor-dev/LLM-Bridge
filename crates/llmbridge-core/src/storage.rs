use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::SystemTime;

/// Source identity snapshot: detects inode/device replacement on Unix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub file_bytes: u64,
    pub modified: Option<SystemTime>,
    #[cfg(unix)]
    pub device: u64,
    #[cfg(unix)]
    pub inode: u64,
}
impl SourceIdentity {
    pub fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Self {
            file_bytes: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        }
    }
}

/// Local range reader with one reusable, preallocated read buffer.
/// Bound applies to its explicit buffer only; OS page cache and process RSS are NOT capped.
pub struct BoundedLocalReader {
    file: File,
    file_len: u64,
    source_identity: SourceIdentity,
    buffer: Box<[u8]>,
    disk_bytes_read: u64,
}

impl BoundedLocalReader {
    pub fn open(
        path: impl AsRef<Path>,
        max_chunk_bytes: usize,
        buffer_budget_bytes: usize,
    ) -> io::Result<Self> {
        let path = path.as_ref();
        let symlink = fs::symlink_metadata(path)?;
        if symlink.file_type().is_symlink() || !symlink.file_type().is_file() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "local storage source must be a regular file, not a symlink",
            ));
        }
        if max_chunk_bytes == 0
            || max_chunk_bytes > 32 * 1024 * 1024
            || max_chunk_bytes > buffer_budget_bytes / 2
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "chunk size exceeds the explicit bounded buffer policy",
            ));
        }
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        // Recheck the path after open: a rename/symlink substitution between
        // symlink_metadata and File::open must not bypass the source policy.
        // Compare the opened descriptor with the final path identity on Unix.
        let path_metadata = fs::symlink_metadata(path)?;
        if path_metadata.file_type().is_symlink() || !path_metadata.is_file()
            || SourceIdentity::from_metadata(&path_metadata)
                != SourceIdentity::from_metadata(&metadata)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "local source changed during open or became a symlink",
            ));
        }
        Ok(Self {
            file,
            file_len: metadata.len(),
            source_identity: SourceIdentity::from_metadata(&metadata),
            buffer: vec![0u8; max_chunk_bytes].into_boxed_slice(),
            disk_bytes_read: 0,
        })
    }

    pub fn file_bytes(&self) -> u64 {
        self.file_len
    }

    pub fn identity(&self) -> &SourceIdentity {
        &self.source_identity
    }

    pub fn allocated_chunk_bytes(&self) -> usize {
        self.buffer.len()
    }

    pub fn disk_bytes_read(&self) -> u64 {
        self.disk_bytes_read
    }

    fn assert_unchanged(&self) -> io::Result<()> {
        let current = self.file.metadata()?;
        if SourceIdentity::from_metadata(&current) != self.source_identity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "source changed while reading; reject stale or mixed tensor bytes",
            ));
        }
        Ok(())
    }

    /// Returns a borrow of the reusable buffer, invalidated at the next mutable operation.
    /// No large file-wide allocation occurs.
    pub fn read_range(&mut self, offset: u64, bytes: usize) -> io::Result<&[u8]> {
        let end = offset
            .checked_add(bytes as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range integer overflow"))?;
        if bytes > self.buffer.len() || end > self.file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "range exceeds file or chunk budget",
            ));
        }
        // Reject accounting overflow before touching the source. Saturating the
        // counter would make an I/O audit appear valid after losing bytes.
        let next_disk_bytes_read = self.disk_bytes_read
            .checked_add(bytes as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor read accounting overflow"))?;
        self.assert_unchanged()?;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut self.buffer[..bytes])?;
        self.assert_unchanged()?;
        self.disk_bytes_read = next_disk_bytes_read;
        Ok(&self.buffer[..bytes])
    }

    /// Hash with cooperative cancellation checked before each bounded read.
    /// A cancelled pass returns Interrupted and never yields a partial digest.
    pub fn sha256_cancellable(&mut self, mut cancelled: impl FnMut() -> bool) -> io::Result<String> {
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        loop {
            if cancelled() {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "model hash cancelled"));
            }
            if offset == self.file_len {
                break;
            }
            let bytes = (self.file_len - offset).min(self.buffer.len() as u64) as usize;
            hash.update(self.read_range(offset, bytes)?);
            offset += bytes as u64;
        }
        Ok(format!("{:x}", hash.finalize()))
    }

    /// Whole-file hash verified with bounded chunks. Does not imply signed manifest validation.
    pub fn sha256(&mut self) -> io::Result<String> {
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        while offset < self.file_len {
            let bytes = (self.file_len - offset).min(self.buffer.len() as u64) as usize;
            hash.update(self.read_range(offset, bytes)?);
            offset += bytes as u64;
        }
        Ok(format!("{:x}", hash.finalize()))
    }

    pub fn verify_sha256(&mut self, expected: &str) -> io::Result<String> {
        if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "expected SHA-256 must be 64 hex digits",
            ));
        }
        let actual = self.sha256()?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SHA-256 mismatch; reject model bytes",
            ));
        }
        Ok(actual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture(bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "llmbridge-range-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }

    #[test]
    fn cancelled_hash_stops_before_next_chunk() {
        let path = fixture(b"abcdef");
        let mut reader = BoundedLocalReader::open(&path, 2, 4).unwrap();
        let mut checks = 0;
        let err = reader.sha256_cancellable(|| {
            checks += 1;
            checks == 2
        }).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
        assert_eq!(reader.disk_bytes_read(), 2);
        assert_eq!(checks, 2);
        let digest = reader.sha256_cancellable(|| false).unwrap();
        assert_eq!(digest.len(), 64);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_counter_overflow_fails_before_disk_io() {
        let path = fixture(b"abcdef");
        let mut reader = BoundedLocalReader::open(&path, 2, 4).unwrap();
        reader.disk_bytes_read = u64::MAX - 1;
        let err = reader.read_range(0, 2).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(reader.disk_bytes_read(), u64::MAX - 1);
        // A rejected read must not disturb the counter or reader usability.
        assert_eq!(reader.read_range(0, 1).unwrap(), b"a");
        assert_eq!(reader.disk_bytes_read(), u64::MAX);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_ranges_only_read_exact_bytes() {
        let path = fixture(b"abcdefghij");
        let mut reader = BoundedLocalReader::open(&path, 3, 8).unwrap();
        assert_eq!(reader.read_range(4, 3).unwrap(), b"efg");
        assert_eq!(reader.disk_bytes_read(), 3);
        assert_eq!(reader.allocated_chunk_bytes(), 3);
        assert!(reader.read_range(4, 4).is_err());
        assert!(reader.read_range(u64::MAX, 1).is_err());
        assert!(reader.read_range(9, 2).is_err());
        fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_link_sources_are_rejected() {
        use std::os::unix::fs::symlink;
        let path = fixture(b"abc");
        let link = path.with_extension("link");
        symlink(&path, &link).unwrap();
        let result = BoundedLocalReader::open(&link, 2, 4);
        assert!(result.is_err());
        fs::remove_file(link).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn streaming_sha256_matches_known_vector() {
        let path = fixture(b"abc");
        let mut reader = BoundedLocalReader::open(&path, 2, 4).unwrap();
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(reader.verify_sha256(digest).unwrap(), digest);
        assert_eq!(reader.disk_bytes_read(), 3);
        assert!(reader.verify_sha256(&"0".repeat(64)).is_err());
        assert!(reader.verify_sha256("missing").is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn changed_source_and_invalid_budget_are_rejected() {
        let path = fixture(b"abc");
        assert!(BoundedLocalReader::open(&path, 3, 4).is_err());
        let mut reader = BoundedLocalReader::open(&path, 2, 4).unwrap();
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"d")
            .unwrap();
        assert!(reader.read_range(0, 1).is_err());
        fs::remove_file(path).unwrap();
    }
}
