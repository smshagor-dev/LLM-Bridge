//! Session-local bounded block cache/read-ahead for a single open model source.
//! Identity checks detect ordinary source changes, not cryptographic authenticity.
use crate::storage::BoundedLocalReader;
use std::io;

#[derive(Clone, Copy, Default)]
struct Slot {
    offset: u64,
    used_at: u64,
    valid: bool,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadCacheMetrics {
    pub allocated_bytes: usize,
    pub block_hits: u64,
    pub block_misses: u64,
    pub evictions: u64,
    pub source_read_calls: u64,
}

/// Owns the source and cache for one generation. Never shared across model sessions.
/// The cache budget covers fixed payload, response staging and slot arrays, excluding
/// allocator headers and the source reader's separately reported buffer.
pub struct CachedLocalReader {
    source: BoundedLocalReader,
    payload: Vec<u8>,
    response: Vec<u8>,
    slots: Vec<Slot>,
    tick: u64,
    metrics: ReadCacheMetrics,
}
fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
fn increment(value: &mut u64) -> io::Result<()> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| invalid("cache counter overflow"))?;
    Ok(())
}
fn allocate<T: Default + Clone>(count: usize) -> io::Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "cache reservation failed"))?;
    values.resize(count, T::default());
    Ok(values)
}
impl CachedLocalReader {
    pub fn new(source: BoundedLocalReader, cache_budget: usize) -> io::Result<Self> {
        if cache_budget > 32 * 1024 * 1024 {
            return Err(invalid("cache budget exceeds 32 MiB policy"));
        }
        let chunk = source.allocated_chunk_bytes();
        let count = cache_budget.saturating_sub(chunk) / (chunk + std::mem::size_of::<Slot>());
        if cache_budget != 0 && count == 0 {
            return Err(invalid("cache budget cannot hold staging and one block"));
        }
        // Bound lookup work/metadata even with tiny chunks from a library caller.
        let count = count.min(512);
        let payload_bytes = count * chunk;
        let staging_bytes = if count == 0 { 0 } else { chunk };
        let allocated_bytes = payload_bytes + staging_bytes + count * std::mem::size_of::<Slot>();
        Ok(Self {
            source,
            payload: allocate(payload_bytes)?,
            response: allocate(staging_bytes)?,
            slots: allocate(count)?,
            tick: 0,
            metrics: ReadCacheMetrics {
                allocated_bytes,
                ..Default::default()
            },
        })
    }
    pub fn metrics(&self) -> ReadCacheMetrics {
        self.metrics
    }
    pub fn source_bytes_read(&self) -> u64 {
        self.source.disk_bytes_read()
    }

    /// Read-ahead fills one aligned source block, never beyond EOF or the reader chunk.
    /// Cache hits revalidate the original descriptor/path identity before publication.
    /// Cancellation or read failure returns no borrowed bytes; partial staging stays private.
    pub fn read_range_cancellable(
        &mut self,
        offset: u64,
        length: usize,
        mut cancelled: impl FnMut() -> bool,
    ) -> io::Result<&[u8]> {
        if self.slots.is_empty() {
            increment(&mut self.metrics.source_read_calls)?;
            return self
                .source
                .read_range_cancellable(offset, length, cancelled);
        }
        let chunk = self.source.allocated_chunk_bytes();
        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| invalid("cached range overflow"))?;
        if length > chunk || end > self.source.file_bytes() {
            return Err(invalid("cached range exceeds source/chunk"));
        }
        if cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "cached read cancelled",
            ));
        }
        self.source.validate_source()?;
        let mut copied = 0;
        while copied < length {
            if cancelled() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "cached read cancelled",
                ));
            }
            increment(&mut self.tick)?;
            let current = offset + copied as u64;
            let block_start = current / chunk as u64 * chunk as u64;
            let within = (current - block_start) as usize;
            let available = (self.source.file_bytes() - block_start).min(chunk as u64) as usize;
            let count = (length - copied).min(available - within);
            let slot = match self
                .slots
                .iter()
                .position(|s| s.valid && s.offset == block_start)
            {
                Some(index) => {
                    increment(&mut self.metrics.block_hits)?;
                    index
                }
                None => {
                    let index = self.slots.iter().position(|s| !s.valid).unwrap_or_else(|| {
                        self.slots
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, s)| s.used_at)
                            .map(|(i, _)| i)
                            .expect("nonempty cache")
                    });
                    increment(&mut self.metrics.block_misses)?;
                    if self.slots[index].valid {
                        increment(&mut self.metrics.evictions)?;
                    }
                    // Invalidate before I/O: a failed refill never exposes the old block.
                    self.slots[index].valid = false;
                    increment(&mut self.metrics.source_read_calls)?;
                    let bytes = self.source.read_range_cancellable(
                        block_start,
                        available,
                        &mut cancelled,
                    )?;
                    self.payload[index * chunk..index * chunk + available].copy_from_slice(bytes);
                    self.slots[index] = Slot {
                        offset: block_start,
                        used_at: self.tick,
                        valid: true,
                    };
                    index
                }
            };
            self.slots[slot].used_at = self.tick;
            let begin = slot * chunk + within;
            self.response[copied..copied + count]
                .copy_from_slice(&self.payload[begin..begin + count]);
            copied += count;
        }
        if cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "cached read cancelled",
            ));
        }
        self.source.validate_source()?;
        Ok(&self.response[..length])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "bridge-read-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, b"abcdefghijklmnopqrstuvwxyz").unwrap();
            Self(path)
        }
        fn reader(&self, slots: usize) -> CachedLocalReader {
            let source = BoundedLocalReader::open(&self.0, 8, 16).unwrap();
            CachedLocalReader::new(
                source,
                if slots == 0 {
                    0
                } else {
                    8 + slots * (8 + std::mem::size_of::<Slot>())
                },
            )
            .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    #[test]
    fn adjacent_rows_share_read_ahead_and_cross_block_eof_reads_are_exact() {
        let file = Fixture::new();
        let mut reader = file.reader(2);
        assert_eq!(
            reader.read_range_cancellable(1, 3, || false).unwrap(),
            b"bcd"
        );
        assert_eq!(
            reader.read_range_cancellable(4, 3, || false).unwrap(),
            b"efg"
        );
        assert_eq!(reader.source_bytes_read(), 8);
        assert_eq!(
            reader.read_range_cancellable(6, 5, || false).unwrap(),
            b"ghijk"
        );
        assert_eq!(reader.source_bytes_read(), 16);
        assert_eq!(
            reader.read_range_cancellable(24, 2, || false).unwrap(),
            b"yz"
        );
        assert_eq!(reader.source_bytes_read(), 18);
        assert!(reader.metrics().block_hits >= 2);
        assert!(reader.read_range_cancellable(24, 3, || false).is_err());
        assert!(reader
            .read_range_cancellable(u64::MAX, 2, || false)
            .is_err());
    }
    #[test]
    fn lru_evicts_oldest_and_keeps_fixed_budget() {
        let file = Fixture::new();
        let mut reader = file.reader(2);
        let allocated = reader.metrics().allocated_bytes;
        for offset in [0, 8, 0, 16, 0, 8] {
            reader.read_range_cancellable(offset, 2, || false).unwrap();
        }
        assert_eq!(reader.metrics().source_read_calls, 4);
        assert_eq!(reader.metrics().evictions, 2);
        assert_eq!(reader.metrics().allocated_bytes, allocated);
    }
    #[test]
    fn cache_hit_rejects_changed_source_instead_of_serving_stale_weights() {
        let file = Fixture::new();
        let mut reader = file.reader(2);
        reader.read_range_cancellable(0, 2, || false).unwrap();
        std::fs::write(&file.0, b"changed-size").unwrap();
        assert!(reader.read_range_cancellable(0, 2, || false).is_err());
        assert_eq!(reader.source_bytes_read(), 8);
    }
    #[test]
    fn cancellation_before_or_after_hit_never_publishes_bytes() {
        let file = Fixture::new();
        let mut reader = file.reader(2);
        reader.read_range_cancellable(0, 2, || false).unwrap();
        let calls = reader.metrics().source_read_calls;
        assert_eq!(
            reader
                .read_range_cancellable(0, 2, || true)
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
        let mut polls = 0;
        assert_eq!(
            reader
                .read_range_cancellable(0, 2, || {
                    polls += 1;
                    polls == 3
                })
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
        assert_eq!(reader.metrics().source_read_calls, calls);
    }
    #[test]
    fn late_source_change_on_cache_hit_is_rejected_before_publication() {
        let file = Fixture::new();
        let mut reader = file.reader(2);
        reader.read_range_cancellable(0, 2, || false).unwrap();
        let mut polls = 0;
        let error = reader
            .read_range_cancellable(0, 2, || {
                polls += 1;
                if polls == 3 {
                    std::fs::write(&file.0, b"replaced-after-cache-copy").unwrap();
                }
                false
            })
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(reader.source_bytes_read(), 8);
    }
    #[test]
    fn cancelled_refill_is_not_reused_as_a_valid_cache_block() {
        let file = Fixture::new();
        let mut reader = file.reader(1);
        reader.read_range_cancellable(0, 2, || false).unwrap();
        let mut polls = 0;
        assert_eq!(
            reader
                .read_range_cancellable(8, 2, || {
                    polls += 1;
                    polls == 4
                })
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
        let after_failure = reader.source_bytes_read();
        assert_eq!(
            reader.read_range_cancellable(8, 2, || false).unwrap(),
            b"ij"
        );
        assert_eq!(reader.source_bytes_read(), after_failure + 8);
    }

    #[test]
    fn disabled_cache_has_identical_ranges_without_read_ahead() {
        let file = Fixture::new();
        let mut reader = file.reader(0);
        assert_eq!(
            reader.read_range_cancellable(1, 3, || false).unwrap(),
            b"bcd"
        );
        assert_eq!(
            reader.read_range_cancellable(4, 3, || false).unwrap(),
            b"efg"
        );
        assert_eq!(reader.source_bytes_read(), 6);
        assert_eq!(reader.metrics().allocated_bytes, 0);
    }
    #[test]
    fn insufficient_or_excessive_cache_budget_is_rejected() {
        let file = Fixture::new();
        for budget in [1, 8, 8 + std::mem::size_of::<Slot>(), 33 * 1024 * 1024] {
            assert!(CachedLocalReader::new(
                BoundedLocalReader::open(&file.0, 8, 16).unwrap(),
                budget
            )
            .is_err());
        }
    }
}
