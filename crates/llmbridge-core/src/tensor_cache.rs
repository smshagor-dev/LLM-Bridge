//! Explicitly budgeted cache for verified tensor chunks. No OS RSS guarantee.
//! Entries are keyed by source identity supplied by the caller, tensor offset and length.
//! A changed source must receive a different identity or the caller must clear the cache.
use std::collections::{HashMap, VecDeque};
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub source_revision: [u8; 32],
    pub file_offset: u64,
    pub length: usize,
}

pub struct TensorChunkCache {
    capacity: usize,
    used: usize,
    entries: HashMap<ChunkKey, Vec<u8>>,
    lru: VecDeque<ChunkKey>,
}

impl TensorChunkCache {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, used: 0, entries: HashMap::new(), lru: VecDeque::new() }
    }

    pub fn used_bytes(&self) -> usize { self.used }
    pub fn capacity_bytes(&self) -> usize { self.capacity }
    pub fn len(&self) -> usize { self.entries.len() }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used = 0;
    }

    pub fn get(&mut self, key: &ChunkKey) -> Option<&[u8]> {
        if !self.entries.contains_key(key) { return None; }
        self.lru.retain(|item| item != key);
        self.lru.push_back(*key);
        self.entries.get(key).map(Vec::as_slice)
    }

    /// Caller must authenticate bytes before insertion; this cache does not verify hashes.
    /// Oversized entries are rejected without modifying existing entries.
    pub fn insert(&mut self, key: ChunkKey, bytes: &[u8]) -> io::Result<()> {
        if key.length == 0 || key.length != bytes.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "cache key length mismatch"));
        }
        if bytes.len() > self.capacity {
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "chunk exceeds cache budget"));
        }
        // Allocate before evicting so a failed allocation leaves existing entries intact.
        let mut owned = Vec::new();
        owned.try_reserve_exact(bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "cache allocation failed"))?;
        owned.extend_from_slice(bytes);
        if let Some(previous) = self.entries.remove(&key) {
            self.used -= previous.len();
            self.lru.retain(|item| item != &key);
        }
        while self.used > self.capacity - owned.len() {
            let oldest = self.lru.pop_front().expect("nonempty cache has LRU entry");
            let removed = self.entries.remove(&oldest).expect("LRU entry exists");
            self.used -= removed.len();
        }
        self.used += owned.len();
        self.entries.insert(key, owned);
        self.lru.push_back(key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(revision: u8, offset: u64, length: usize) -> ChunkKey {
        ChunkKey { source_revision: [revision; 32], file_offset: offset, length }
    }

    #[test]
    fn evicts_least_recently_used_and_never_exceeds_budget() {
        let mut cache = TensorChunkCache::new(6);
        cache.insert(key(1, 0, 3), b"abc").unwrap();
        cache.insert(key(1, 3, 3), b"def").unwrap();
        assert_eq!(cache.get(&key(1, 0, 3)), Some(&b"abc"[..]));
        cache.insert(key(1, 6, 3), b"ghi").unwrap();
        assert!(cache.get(&key(1, 3, 3)).is_none());
        assert_eq!(cache.get(&key(1, 0, 3)), Some(&b"abc"[..]));
        assert_eq!(cache.used_bytes(), 6);
    }

    #[test]
    fn source_revision_isolation_and_replacement() {
        let mut cache = TensorChunkCache::new(4);
        cache.insert(key(1, 0, 2), b"aa").unwrap();
        assert!(cache.get(&key(2, 0, 2)).is_none());
        cache.insert(key(1, 0, 2), b"bb").unwrap();
        assert_eq!(cache.used_bytes(), 2);
        assert_eq!(cache.get(&key(1, 0, 2)), Some(&b"bb"[..]));
        cache.clear();
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn invalid_and_oversized_insertions_do_not_evict() {
        let mut cache = TensorChunkCache::new(2);
        cache.insert(key(1, 0, 2), b"ok").unwrap();
        assert!(cache.insert(key(1, 2, 3), b"bad").is_err());
        assert!(cache.insert(key(1, 2, 1), b"bad").is_err());
        assert_eq!(cache.get(&key(1, 0, 2)), Some(&b"ok"[..]));
        assert_eq!(cache.used_bytes(), 2);
    }

    #[test]
    fn zero_budget_never_caches() {
        let mut cache = TensorChunkCache::new(0);
        assert!(cache.insert(key(1, 0, 1), b"x").is_err());
        assert_eq!(cache.len(), 0);
    }
}
