//! Explicitly budgeted cache for verified tensor chunks. No OS RSS guarantee.
//! Entries are keyed by source identity supplied by the caller, tensor offset and length.
//! A changed source must receive a different identity or the caller must clear the cache.
use std::collections::{HashMap, VecDeque};
use std::io;
use sha2::{Digest, Sha256};

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
        Self {
            capacity,
            used: 0,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub fn used_bytes(&self) -> usize {
        self.used
    }
    pub fn capacity_bytes(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Change the logical chunk-cache budget at runtime. Evict least-recently
    /// used chunks immediately when shrinking; never retain bytes over budget.
    /// This does not limit process RSS or the allocator's retained pages.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        while self.used > self.capacity {
            let oldest = self.lru.pop_front().expect("nonempty cache has LRU entry");
            let removed = self.entries.remove(&oldest).expect("LRU entry exists");
            self.used -= removed.len();
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used = 0;
    }

    /// Reclaim chunks belonging to previous authenticated model revisions.
    /// The caller must supply a trusted revision digest, not a filename.
    pub fn retain_revision(&mut self, revision: [u8; 32]) {
        self.entries
            .retain(|key, _| key.source_revision == revision);
        self.lru.retain(|key| key.source_revision == revision);
        self.used = self.entries.values().map(Vec::len).sum();
    }

    pub fn get(&mut self, key: &ChunkKey) -> Option<&[u8]> {
        if !self.entries.contains_key(key) {
            return None;
        }
        self.lru.retain(|item| item != key);
        self.lru.push_back(*key);
        self.entries.get(key).map(Vec::as_slice)
    }

    /// Authenticate the exact chunk bytes against an independently trusted SHA-256
    /// before changing the cache. The model revision in ChunkKey must separately
    /// identify a trusted source; this digest alone does not authenticate a model.
    pub fn insert_verified(
        &mut self,
        key: ChunkKey,
        bytes: &[u8],
        expected_sha256: &[u8; 32],
    ) -> io::Result<()> {
        if key.length == 0 || key.length != bytes.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "cache key length mismatch"));
        }
        let digest = Sha256::digest(bytes);
        if digest.as_slice() != expected_sha256 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tensor chunk SHA-256 mismatch"));
        }
        self.insert(key, bytes)
    }

    /// Caller must authenticate bytes before insertion; this cache does not verify hashes.
    /// Oversized entries are rejected without modifying existing entries.
    pub fn insert(&mut self, key: ChunkKey, bytes: &[u8]) -> io::Result<()> {
        if key.length == 0 || key.length != bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cache key length mismatch",
            ));
        }
        // Reject overlapping ranges for one authenticated revision: otherwise
        // distinct keys could return conflicting bytes for the same tensor region.
        let end = key.file_offset.checked_add(key.length as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "cache range offset overflow")
        })?;
        for existing in self.entries.keys() {
            if existing.source_revision != key.source_revision || *existing == key {
                continue;
            }
            let existing_end = existing.file_offset.checked_add(existing.length as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid cached range"))?;
            if key.file_offset < existing_end && existing.file_offset < end {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "overlapping cached tensor ranges"));
            }
        }
        if bytes.len() > self.capacity {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "chunk exceeds cache budget",
            ));
        }
        // Allocate before evicting so a failed allocation leaves existing entries intact.
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "cache allocation failed"))?;
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
        ChunkKey {
            source_revision: [revision; 32],
            file_offset: offset,
            length,
        }
    }

    #[test]
    fn verified_insert_rejects_bad_hash_without_evicting_existing_chunks() {
        let mut cache = TensorChunkCache::new(4);
        cache.insert(key(1, 0, 4), b"safe").unwrap();
        let correct: [u8; 32] = Sha256::digest(b"next").into();
        let wrong = [0u8; 32];
        assert_eq!(
            cache.insert_verified(key(1, 4, 4), b"next", &wrong)
                .unwrap_err().kind(), io::ErrorKind::InvalidData
        );
        assert_eq!(cache.get(&key(1, 0, 4)), Some(&b"safe"[..]));
        assert_eq!(cache.used_bytes(), 4);
        cache.insert_verified(key(1, 4, 4), b"next", &correct).unwrap();
        assert_eq!(cache.get(&key(1, 4, 4)), Some(&b"next"[..]));
        assert_eq!(cache.used_bytes(), 4);
        assert!(cache.get(&key(1, 0, 4)).is_none());
    }

    #[test]
    fn verified_replacement_mismatch_preserves_old_value() {
        let mut cache = TensorChunkCache::new(4);
        cache.insert(key(1, 0, 4), b"good").unwrap();
        let digest: [u8; 32] = Sha256::digest(b"good").into();
        assert_eq!(
            cache.insert_verified(key(1, 0, 4), b"evil", &digest)
                .unwrap_err().kind(), io::ErrorKind::InvalidData
        );
        assert_eq!(cache.get(&key(1, 0, 4)), Some(&b"good"[..]));
    }

    #[test]
    fn shrinking_budget_evicts_lru_and_preserves_recent_chunks() {
        let mut cache = TensorChunkCache::new(9);
        cache.insert(key(1, 0, 3), b"old").unwrap();
        cache.insert(key(1, 3, 3), b"mid").unwrap();
        cache.insert(key(1, 6, 3), b"new").unwrap();
        assert_eq!(cache.get(&key(1, 0, 3)), Some(&b"old"[..]));
        cache.set_capacity(6);
        assert_eq!(cache.used_bytes(), 6);
        assert!(cache.get(&key(1, 3, 3)).is_none());
        assert_eq!(cache.get(&key(1, 0, 3)), Some(&b"old"[..]));
        assert_eq!(cache.get(&key(1, 6, 3)), Some(&b"new"[..]));
        cache.set_capacity(0);
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.len(), 0);
        assert!(cache.insert(key(1, 9, 1), b"x").is_err());
        cache.set_capacity(2);
        cache.insert(key(1, 9, 2), b"ok").unwrap();
        assert_eq!(cache.used_bytes(), 2);
    }

    #[test]
    fn growing_budget_preserves_entries_and_accounting() {
        let mut cache = TensorChunkCache::new(2);
        cache.insert(key(1, 0, 2), b"ab").unwrap();
        cache.set_capacity(4);
        cache.insert(key(1, 2, 2), b"cd").unwrap();
        assert_eq!(cache.used_bytes(), 4);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&key(1, 0, 2)), Some(&b"ab"[..]));
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
    fn overlapping_ranges_are_rejected_without_evicting() {
        let mut cache = TensorChunkCache::new(6);
        cache.insert(key(1, 0, 3), b"abc").unwrap();
        cache.insert(key(1, 3, 3), b"def").unwrap();
        assert_eq!(
            cache.insert(key(1, 2, 3), b"xyz").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(cache.used_bytes(), 6);
        assert_eq!(cache.get(&key(1, 0, 3)), Some(&b"abc"[..]));
        assert_eq!(cache.get(&key(1, 3, 3)), Some(&b"def"[..]));
        cache.insert(key(2, 0, 3), b"new").unwrap();
        assert_eq!(cache.get(&key(2, 0, 3)), Some(&b"new"[..]));
    }

    #[test]
    fn overflowing_file_ranges_fail_closed_and_preserve_cache() {
        let mut cache = TensorChunkCache::new(4);
        cache.insert(key(1, 0, 2), b"ok").unwrap();
        assert_eq!(
            cache.insert(key(1, u64::MAX, 2), b"no").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(cache.get(&key(1, 0, 2)), Some(&b"ok"[..]));
        assert_eq!(cache.used_bytes(), 2);
    }

    #[test]
    fn zero_budget_never_caches() {
        let mut cache = TensorChunkCache::new(0);
        assert!(cache.insert(key(1, 0, 1), b"x").is_err());
        assert_eq!(cache.len(), 0);
    }
    #[test]
    fn model_switch_reclaims_old_revision_and_preserves_new_entries() {
        let mut cache = TensorChunkCache::new(8);
        cache.insert(key(1, 0, 3), b"old").unwrap();
        cache.insert(key(2, 3, 2), b"ab").unwrap();
        cache.insert(key(2, 5, 2), b"cd").unwrap();
        cache.retain_revision([2; 32]);
        assert_eq!(cache.used_bytes(), 4);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&key(1, 0, 3)).is_none());
        cache.insert(key(2, 7, 4), b"efgh").unwrap();
        assert_eq!(cache.used_bytes(), 8);
        assert_eq!(cache.get(&key(2, 3, 2)), Some(&b"ab"[..]));
        cache.retain_revision([9; 32]);
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(cache.len(), 0);
    }
}
