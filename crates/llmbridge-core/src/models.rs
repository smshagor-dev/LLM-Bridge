use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub path: PathBuf,
    pub file_bytes: u64,
    pub format_version: u32,
    pub tensor_count: u64,
    pub metadata_entries: u64,
}
/// Strict bounded header inspection, NOT tensor or execution verification.
pub fn inspect_gguf(path: impl AsRef<Path>) -> io::Result<GgufHeader> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len < 24 { return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated GGUF header")); }
    let mut h = [0u8; 24];
    file.read_exact(&mut h)?;
    if &h[0..4] != b"GGUF" { return Err(io::Error::new(io::ErrorKind::InvalidData, "not GGUF")); }
    let version = u32::from_le_bytes(h[4..8].try_into().unwrap());
    if !(2..=3).contains(&version) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "GGUF version unsupported"));
    }
    let tensor_count = u64::from_le_bytes(h[8..16].try_into().unwrap());
    let metadata_entries = u64::from_le_bytes(h[16..24].try_into().unwrap());
    if tensor_count == 0 || tensor_count > 10_000_000 || metadata_entries > 1_000_000 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid GGUF counts"));
    }
    Ok(GgufHeader { path: path.to_owned(), file_bytes: len, format_version: version, tensor_count, metadata_entries })
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    fn fixture(bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("llmbridge-{}-{}",std::process::id(),N.fetch_add(1,Ordering::Relaxed)));
        File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }
    #[test]
    fn inspect_only_fixed_size_header() {
        let mut b=b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&11u64.to_le_bytes());
        b.extend_from_slice(&5u64.to_le_bytes());
        let p=fixture(&b);
        let h=inspect_gguf(&p).unwrap();
        assert_eq!((h.format_version,h.tensor_count,h.metadata_entries),(3,11,5));
        std::fs::remove_file(p).unwrap();
    }
    #[test]
    fn invalid_header_is_rejected() {
        for b in [b"GG".as_slice(),b"INVALID_FORMAT".as_slice()] {
            let p=fixture(b);
            assert!(inspect_gguf(&p).is_err());
            std::fs::remove_file(p).unwrap();
        }
    }
    #[test]
    fn impossible_counts_are_rejected() {
        let mut b=b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let p=fixture(&b);
        assert!(inspect_gguf(&p).is_err());
        std::fs::remove_file(p).unwrap();
    }
}
