//! Opt-in, bounded exact-range local lookahead. Callers must schedule only
//! ranges belonging to the current tensor and consume them in order.
use crate::storage::BoundedLocalReader;
use std::any::Any;
use std::io;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
    Arc,
};
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// Minimum requested stack for the bounded blocking reader worker.
pub const MIN_LOOKAHEAD_WORKER_STACK_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct LookaheadLimits {
    /// Total admitted memory, including fixed worker allowances below.
    pub staging_bytes: usize,
    pub max_outstanding_reads: usize,
    /// Reserved allowance for one reader buffer, bounded channels, request
    /// descriptors and worker bookkeeping. Must be supplied by the caller.
    pub fixed_overhead_bytes: usize,
    /// Also used as the requested OS worker stack size.
    pub worker_stack_bytes: usize,
}
struct Request {
    offset: u64,
    length: usize,
    tensor_start: u64,
    tensor_end: u64,
    deadline: Option<Instant>,
    queued_at: Instant,
    lease: Lease,
}
struct ResultBytes {
    bytes: Vec<u8>,
    lease: Lease,
}
#[derive(Debug)]
struct Lease {
    bytes: usize,
    capacity: usize,
    reserved: Arc<AtomicUsize>,
    outstanding: Arc<AtomicUsize>,
    actual_staging: Arc<AtomicUsize>,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
        self.outstanding.fetch_sub(1, Ordering::AcqRel);
        self.actual_staging
            .fetch_sub(self.capacity, Ordering::AcqRel);
    }
}
/// A completed result retains its staging reservation until it is dropped.
#[derive(Debug)]
pub struct PrefetchedRange {
    pub offset: u64,
    bytes: Vec<u8>,
    _lease: Lease,
}
impl PrefetchedRange {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// One worker and a bounded command/result channel. A single request may be in
/// service and at most `max_outstanding_reads` reservations may exist, including
/// completed buffers retained by the consumer. No tensor/model-sized buffers.
pub struct LocalLookahead {
    tx: Option<SyncSender<Request>>,
    rx: Receiver<io::Result<(u64, ResultBytes)>>,
    join: Option<JoinHandle<()>>,
    cancelled: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    reserved: Arc<AtomicUsize>,
    outstanding: Arc<AtomicUsize>,
    byte_limit: usize,
    max_reads: usize,
    chunk: usize,
    base_reserved: usize,
    source_bytes: Arc<AtomicU64>,
    queue_wait_ns: Arc<AtomicU64>,
    worker_read_ns: Arc<AtomicU64>,
    staging_copy_ns: Arc<AtomicU64>,
    consumer_wait_ns: Arc<AtomicU64>,
    actual_staging: Arc<AtomicUsize>,
    peak_actual_staging: Arc<AtomicUsize>,
}
#[derive(Debug, Clone, Default)]
pub struct LookaheadFinish {
    pub worker_terminated: bool,
    pub reserved_bytes: usize,
    pub outstanding_reads: usize,
    pub source_bytes_read: u64,
    pub worker_panic: Option<String>,
    pub queue_wait_ns: u64,
    pub worker_read_ns: u64,
    pub staging_copy_ns: u64,
    pub consumer_wait_ns: u64,
    pub peak_actual_staging_bytes: usize,
}
impl LocalLookahead {
    pub fn start(reader: BoundedLocalReader, limits: LookaheadLimits) -> io::Result<Self> {
        Self::start_internal(reader, limits, None)
    }
    #[cfg(test)]
    fn start_with_test_hook(
        reader: BoundedLocalReader,
        limits: LookaheadLimits,
        hook: Box<dyn Fn() -> io::Result<()> + Send>,
    ) -> io::Result<Self> {
        Self::start_internal(reader, limits, Some(hook))
    }
    fn start_internal(
        reader: BoundedLocalReader,
        limits: LookaheadLimits,
        before_read: Option<Box<dyn Fn() -> io::Result<()> + Send>>,
    ) -> io::Result<Self> {
        let chunk = reader.allocated_chunk_bytes();
        if limits.worker_stack_bytes < MIN_LOOKAHEAD_WORKER_STACK_BYTES {
            return Err(invalid("lookahead worker stack allowance is below 1 MiB"));
        }
        if limits.staging_bytes == 0
            || limits.max_outstanding_reads == 0
            || chunk == 0
            || limits.fixed_overhead_bytes == 0
        {
            return Err(invalid("lookahead limits must be nonzero"));
        }
        let queue = limits.max_outstanding_reads.min(8);
        let channel_bytes = queue
            .checked_mul(
                std::mem::size_of::<Request>()
                    + std::mem::size_of::<io::Result<(u64, ResultBytes)>>(),
            )
            .ok_or_else(|| invalid("lookahead channel accounting overflow"))?;
        let fixed = chunk
            .checked_add(channel_bytes)
            .and_then(|n| n.checked_add(limits.worker_stack_bytes))
            .ok_or_else(|| invalid("lookahead fixed accounting overflow"))?;
        if fixed > limits.fixed_overhead_bytes || fixed > limits.staging_bytes {
            return Err(invalid("lookahead budget does not cover worker overhead"));
        }
        let (tx, requests) = mpsc::sync_channel::<Request>(queue);
        let (results_tx, rx) = mpsc::sync_channel(queue);
        let cancelled = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        // Reserve modeled worker/channel/source overhead before creating the
        // thread or allowing any request buffer allocation.
        let reserved = Arc::new(AtomicUsize::new(limits.fixed_overhead_bytes));
        let outstanding = Arc::new(AtomicUsize::new(0));
        let source_bytes = Arc::new(AtomicU64::new(0));
        let queue_wait_ns = Arc::new(AtomicU64::new(0));
        let worker_read_ns = Arc::new(AtomicU64::new(0));
        let staging_copy_ns = Arc::new(AtomicU64::new(0));
        let consumer_wait_ns = Arc::new(AtomicU64::new(0));
        let actual_staging = Arc::new(AtomicUsize::new(0));
        let peak_actual_staging = Arc::new(AtomicUsize::new(0));
        let worker_actual_staging = Arc::clone(&actual_staging);
        let worker_peak_staging = Arc::clone(&peak_actual_staging);
        let worker_source_bytes = Arc::clone(&source_bytes);
        let worker_queue_wait = Arc::clone(&queue_wait_ns);
        let worker_read_time = Arc::clone(&worker_read_ns);
        let worker_copy_time = Arc::clone(&staging_copy_ns);
        let worker_cancel = Arc::clone(&cancelled);
        let worker_failed = Arc::clone(&failed);
        let join = thread::Builder::new()
            .name("llmbridge-local-read".into())
            .stack_size(limits.worker_stack_bytes)
            .spawn(move || {
                let mut reader = reader;
                let before_read = before_read;
                while let Ok(mut req) = requests.recv() {
                    add_nanos(&worker_queue_wait, req.queued_at.elapsed());
                    let result = (|| {
                        let end = req
                            .offset
                            .checked_add(req.length as u64)
                            .ok_or_else(|| invalid("lookahead range overflow"))?;
                        if req.length == 0
                            || req.length > chunk
                            || req.offset < req.tensor_start
                            || end > req.tensor_end
                        {
                            return Err(invalid("lookahead range crosses tensor/chunk boundary"));
                        }
                        let mut bytes = Vec::new();
                        bytes.try_reserve_exact(req.length).map_err(|_| {
                            io::Error::new(
                                io::ErrorKind::OutOfMemory,
                                "lookahead staging allocation failed",
                            )
                        })?;
                        bytes.resize(req.length, 0);
                        req.lease.capacity = bytes.capacity();
                        let now = worker_actual_staging
                            .fetch_add(req.lease.capacity, Ordering::AcqRel)
                            .saturating_add(req.lease.capacity);
                        worker_peak_staging.fetch_max(now, Ordering::AcqRel);
                        if let Some(hook) = before_read.as_ref() {
                            hook()?;
                        }
                        if worker_cancel.load(Ordering::Acquire) {
                            return Err(io::Error::new(
                                io::ErrorKind::Interrupted,
                                "lookahead cancelled before read",
                            ));
                        }
                        let deadline = req.deadline;
                        let read_started = Instant::now();
                        let read_result =
                            reader.read_range_cancellable(req.offset, req.length, || {
                                worker_cancel.load(Ordering::Acquire)
                                    || deadline.is_some_and(|d| Instant::now() >= d)
                            });
                        add_nanos(&worker_read_time, read_started.elapsed());
                        let slice = read_result?;
                        let copy_started = Instant::now();
                        bytes.copy_from_slice(slice);
                        add_nanos(&worker_copy_time, copy_started.elapsed());
                        worker_source_bytes.store(reader.disk_bytes_read(), Ordering::Release);
                        if worker_cancel.load(Ordering::Acquire)
                            || deadline.is_some_and(|d| Instant::now() >= d)
                        {
                            return Err(io::Error::new(
                                io::ErrorKind::Interrupted,
                                "lookahead cancelled or deadline expired",
                            ));
                        }
                        Ok((
                            req.offset,
                            ResultBytes {
                                bytes,
                                lease: req.lease,
                            },
                        ))
                    })();
                    if result.is_err() {
                        worker_failed.store(true, Ordering::Release);
                    }
                    if results_tx.send(result).is_err() {
                        break;
                    }
                    if worker_cancel.load(Ordering::Acquire)
                        || worker_failed.load(Ordering::Acquire)
                    {
                        break;
                    }
                }
            })
            .map_err(|e| io::Error::other(e.to_string()))?;
        Ok(Self {
            tx: Some(tx),
            rx,
            join: Some(join),
            cancelled,
            failed,
            reserved,
            outstanding,
            byte_limit: limits.staging_bytes,
            max_reads: limits.max_outstanding_reads,
            chunk,
            base_reserved: limits.fixed_overhead_bytes,
            source_bytes,
            queue_wait_ns,
            worker_read_ns,
            staging_copy_ns,
            consumer_wait_ns,
            actual_staging,
            peak_actual_staging,
        })
    }
    /// Admission is charged before the request is queued and before worker allocation.
    pub fn schedule(
        &self,
        offset: u64,
        length: usize,
        tensor_start: u64,
        tensor_bytes: u64,
        deadline: Option<Instant>,
    ) -> io::Result<()> {
        if self.failed.load(Ordering::Acquire) || self.cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "lookahead stopped after prior failure",
            ));
        }
        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| invalid("lookahead range overflow"))?;
        let tensor_end = tensor_start
            .checked_add(tensor_bytes)
            .ok_or_else(|| invalid("tensor range overflow"))?;
        if length == 0 || length > self.chunk || offset < tensor_start || end > tensor_end {
            return Err(invalid("lookahead range crosses tensor/chunk boundary"));
        }
        let mut old = self.reserved.load(Ordering::Acquire);
        loop {
            let next = old
                .checked_add(length)
                .ok_or_else(|| invalid("lookahead accounting overflow"))?;
            if next > self.byte_limit {
                return Err(invalid("insufficient lookahead staging admission"));
            }
            match self.reserved.compare_exchange_weak(
                old,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(now) => old = now,
            }
        }
        let mut active = self.outstanding.load(Ordering::Acquire);
        loop {
            if active >= self.max_reads {
                self.reserved.fetch_sub(length, Ordering::AcqRel);
                return Err(invalid("lookahead outstanding-read limit reached"));
            }
            let next = active
                .checked_add(1)
                .ok_or_else(|| invalid("lookahead read counter overflow"))?;
            match self.outstanding.compare_exchange_weak(
                active,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(now) => active = now,
            }
        }
        let request = Request {
            offset,
            length,
            tensor_start,
            tensor_end,
            deadline,
            queued_at: Instant::now(),
            lease: Lease {
                bytes: length,
                capacity: 0,
                reserved: Arc::clone(&self.reserved),
                outstanding: Arc::clone(&self.outstanding),
                actual_staging: Arc::clone(&self.actual_staging),
            },
        };
        let sent = self
            .tx
            .as_ref()
            .ok_or_else(|| invalid("lookahead is closed"))?
            .try_send(request);
        if let Err(e) = sent {
            return Err(match e {
                mpsc::TrySendError::Full(request) => {
                    drop(request);
                    invalid("lookahead queue is full")
                }
                mpsc::TrySendError::Disconnected(request) => {
                    drop(request);
                    io::Error::new(io::ErrorKind::BrokenPipe, "lookahead worker stopped")
                }
            });
        }
        Ok(())
    }
    pub fn take(&self, cancelled: impl Fn() -> bool) -> io::Result<PrefetchedRange> {
        let waiting_since = Instant::now();
        loop {
            if cancelled() || self.cancelled.load(Ordering::Acquire) {
                self.cancelled.store(true, Ordering::Release);
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "lookahead consumer cancelled",
                ));
            }
            // Block on the channel notification rather than parking the current
            // thread after a poll. The timeout preserves bounded cancellation
            // and deadline observation while a result wakes this consumer at
            // once instead of waiting for the polling interval to elapse.
            match self.rx.recv_timeout(std::time::Duration::from_millis(2)) {
                Ok(Ok((offset, value))) => {
                    if cancelled() || self.cancelled.load(Ordering::Acquire) {
                        self.cancelled.store(true, Ordering::Release);
                        return Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "lookahead consumer cancelled",
                        ));
                    }
                    add_nanos(&self.consumer_wait_ns, waiting_since.elapsed());
                    return Ok(PrefetchedRange {
                        offset,
                        bytes: value.bytes,
                        _lease: value.lease,
                    });
                }
                Ok(Err(error)) => {
                    self.failed.store(true, Ordering::Release);
                    return Err(error);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "lookahead worker stopped",
                    ))
                }
                Err(RecvTimeoutError::Timeout) => {
                    if cancelled() || self.cancelled.load(Ordering::Acquire) {
                        self.cancelled.store(true, Ordering::Release);
                        return Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "lookahead consumer cancelled",
                        ));
                    }
                }
            }
        }
    }
    /// Wait for one result and refuse to publish it unless it matches the
    /// exact range requested by the native tensor-row consumer. This prevents
    /// a stale/late result from being consumed for a different row batch.
    pub fn take_exact(
        &self,
        expected_offset: u64,
        expected_length: usize,
        cancelled: impl Fn() -> bool,
    ) -> io::Result<PrefetchedRange> {
        let result = self.take(cancelled)?;
        if result.offset != expected_offset || result.bytes.len() != expected_length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "prefetched range does not match requested tensor batch",
            ));
        }
        Ok(result)
    }
    pub fn reserved_bytes(&self) -> usize {
        self.total_reserved_bytes()
            .saturating_sub(self.base_reserved)
    }
    pub fn total_reserved_bytes(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }
    pub fn outstanding_reads(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }
    pub fn source_bytes_read(&self) -> u64 {
        self.source_bytes.load(Ordering::Acquire)
    }
    /// Stop submissions, wait for any blocking OS read to return, then release
    /// queued/completed buffers and fixed worker reservations.
    pub fn shutdown(&mut self) -> LookaheadFinish {
        self.cancelled.store(true, Ordering::Release);
        self.tx.take();
        let mut worker_panic = None;
        if let Some(join) = self.join.take() {
            // A full bounded result channel must not deadlock join when the
            // worker is blocked publishing its final completion/error.
            while !join.is_finished() {
                while let Ok(result) = self.rx.try_recv() {
                    drop(result);
                }
                thread::park_timeout(std::time::Duration::from_millis(1));
            }
            if let Err(payload) = join.join() {
                worker_panic = Some(panic_message(payload));
            }
        }
        while let Ok(result) = self.rx.try_recv() {
            drop(result);
        }
        self.reserved
            .fetch_sub(self.base_reserved, Ordering::AcqRel);
        LookaheadFinish {
            worker_terminated: true,
            reserved_bytes: self.total_reserved_bytes(),
            outstanding_reads: self.outstanding_reads(),
            source_bytes_read: self.source_bytes_read(),
            worker_panic,
            queue_wait_ns: self.queue_wait_ns.load(Ordering::Acquire),
            worker_read_ns: self.worker_read_ns.load(Ordering::Acquire),
            staging_copy_ns: self.staging_copy_ns.load(Ordering::Acquire),
            consumer_wait_ns: self.consumer_wait_ns.load(Ordering::Acquire),
            peak_actual_staging_bytes: self.peak_actual_staging.load(Ordering::Acquire),
        }
    }
}
fn add_nanos(counter: &AtomicU64, duration: std::time::Duration) {
    let nanos = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
        Some(old.saturating_add(nanos))
    });
}
fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else {
        "non-string worker panic payload".to_owned()
    }
}
impl Drop for LocalLookahead {
    fn drop(&mut self) {
        if self.join.is_some() {
            let _ = self.shutdown();
        }
    }
}
fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
/// Minimum fixed reservation for the worker's source buffer, bounded channel
/// slots and configured stack. Allocator/kernel bookkeeping is excluded.
pub fn minimum_fixed_overhead_bytes(
    chunk_bytes: usize,
    max_outstanding_reads: usize,
    worker_stack_bytes: usize,
) -> io::Result<usize> {
    if chunk_bytes == 0
        || max_outstanding_reads == 0
        || worker_stack_bytes < MIN_LOOKAHEAD_WORKER_STACK_BYTES
    {
        return Err(invalid(
            "invalid limits or worker stack allowance below 1 MiB",
        ));
    }
    let queue = max_outstanding_reads.min(8);
    let channel_bytes = queue
        .checked_mul(
            std::mem::size_of::<Request>() + std::mem::size_of::<io::Result<(u64, ResultBytes)>>(),
        )
        .ok_or_else(|| invalid("lookahead channel accounting overflow"))?;
    chunk_bytes
        .checked_add(channel_bytes)
        .and_then(|n| n.checked_add(worker_stack_bytes))
        .ok_or_else(|| invalid("lookahead fixed accounting overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        io::Write,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };
    const TEST_FIXED_OVERHEAD: usize = 2 * 1024 * 1024;
    const TEST_WORKER_STACK: usize = MIN_LOOKAHEAD_WORKER_STACK_BYTES;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "llmbridge-lookahead-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::File::create(&path)
                .unwrap()
                .write_all(b"abcdefghijklmnopqrstuvwxyz012345")
                .unwrap();
            Self(path)
        }
        fn reader(&self) -> BoundedLocalReader {
            BoundedLocalReader::open(&self.0, 8, 16).unwrap()
        }
        fn start(&self, bytes: usize, reads: usize) -> LocalLookahead {
            LocalLookahead::start(
                self.reader(),
                LookaheadLimits {
                    staging_bytes: bytes + TEST_FIXED_OVERHEAD,
                    max_outstanding_reads: reads,
                    fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                    worker_stack_bytes: TEST_WORKER_STACK,
                },
            )
            .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    #[test]
    fn exact_budget_and_outstanding_limits_include_retained_results() {
        let f = Fixture::new();
        let worker = f.start(8, 1);
        worker.schedule(0, 8, 0, 16, None).unwrap();
        let range = worker.take(|| false).unwrap();
        assert_eq!(range.bytes(), b"abcdefgh");
        assert_eq!(worker.reserved_bytes(), 8);
        assert!(worker.schedule(8, 1, 0, 16, None).is_err());
        drop(range);
        assert_eq!(worker.reserved_bytes(), 0);
        worker.schedule(8, 8, 0, 16, None).unwrap();
        let second = worker.take(|| false).unwrap();
        assert_eq!(second.bytes(), b"ijklmnop");
    }
    #[test]
    fn exact_range_mismatch_refuses_result_and_releases_its_lease() {
        let f = Fixture::new();
        let worker = f.start(8, 1);
        worker.schedule(0, 8, 0, 16, None).unwrap();
        assert_eq!(
            worker.take_exact(8, 8, || false).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(worker.reserved_bytes(), 0);
        drop(worker);
    }
    #[test]
    fn undersized_worker_stack_is_rejected_before_thread_creation() {
        let f = Fixture::new();
        let error = LocalLookahead::start(
            f.reader(),
            LookaheadLimits {
                staging_bytes: 2 * 1024 * 1024,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: 2 * 1024 * 1024,
                worker_stack_bytes: 2048,
            },
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("below 1 MiB"));
    }
    #[test]
    fn bounded_channels_saturate_without_unbounded_requests_and_shutdown_drains() {
        let f = Fixture::new();
        let entered = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let mut worker = LocalLookahead::start_with_test_hook(
            f.reader(),
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 32,
                max_outstanding_reads: 16,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
            Box::new(move || {
                worker_entered.wait();
                worker_release.wait();
                Ok(())
            }),
        )
        .unwrap();
        worker.schedule(0, 1, 0, 16, None).unwrap();
        entered.wait(); // hold the sole worker in-flight; the request queue has capacity 8
        for _ in 0..8 {
            worker.schedule(0, 1, 0, 16, None).unwrap();
        }
        assert_eq!(
            worker.schedule(0, 1, 0, 16, None).unwrap_err().to_string(),
            "lookahead queue is full"
        );
        release.wait();
        let finish = worker.shutdown(); // joins while draining any bounded result queue
        assert!(finish.worker_terminated);
        assert_eq!(finish.reserved_bytes, 0);
        assert_eq!(finish.outstanding_reads, 0);
    }
    #[test]
    fn pending_slow_read_cancellation_joins_and_releases_all_reservations() {
        let f = Fixture::new();
        let entered = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let mut worker = LocalLookahead::start_with_test_hook(
            f.reader(),
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 8,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
            Box::new(move || {
                worker_entered.wait();
                worker_release.wait();
                Ok(())
            }),
        )
        .unwrap();
        worker.schedule(0, 8, 0, 16, None).unwrap();
        entered.wait(); // prove the worker is held before the OS read
        worker.cancelled.store(true, Ordering::Release);
        release.wait();
        let finish = worker.shutdown();
        assert!(finish.worker_terminated);
        assert_eq!(finish.reserved_bytes, 0);
        assert_eq!(finish.outstanding_reads, 0);
    }
    #[test]
    fn injected_read_error_is_preserved_and_worker_is_joined() {
        let f = Fixture::new();
        let mut worker = LocalLookahead::start_with_test_hook(
            f.reader(),
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 8,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
            Box::new(|| Err(io::Error::other("injected slow-device read failure"))),
        )
        .unwrap();
        worker.schedule(0, 8, 0, 16, None).unwrap();
        let error = worker.take(|| false).unwrap_err();
        assert_eq!(error.to_string(), "injected slow-device read failure");
        let finish = worker.shutdown();
        assert!(finish.worker_terminated);
        assert_eq!(finish.reserved_bytes, 0);
        assert_eq!(finish.outstanding_reads, 0);
    }
    #[test]
    fn worker_panic_is_reported_after_join_and_releases_reservations() {
        let f = Fixture::new();
        let mut worker = LocalLookahead::start_with_test_hook(
            f.reader(),
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 8,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
            Box::new(|| panic!("injected worker panic")),
        )
        .unwrap();
        worker.schedule(0, 8, 0, 16, None).unwrap();
        assert_eq!(
            worker.take(|| false).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        let finish = worker.shutdown();
        assert!(finish.worker_terminated);
        assert_eq!(
            finish.worker_panic.as_deref(),
            Some("injected worker panic")
        );
        assert_eq!(finish.reserved_bytes, 0);
        assert_eq!(finish.outstanding_reads, 0);
    }
    #[test]
    fn insufficient_budget_and_tensor_crossing_reject_before_scheduling() {
        let f = Fixture::new();
        let worker = f.start(7, 2);
        assert!(worker.schedule(0, 8, 0, 16, None).is_err());
        assert!(worker.schedule(12, 8, 0, 16, None).is_err());
        assert_eq!(worker.reserved_bytes(), 0);
    }
    #[test]
    fn cancelled_consumer_joins_worker_and_releases_pending_reservation() {
        let f = Fixture::new();
        let worker = f.start(8, 1);
        let reserved = Arc::clone(&worker.reserved);
        let outstanding = Arc::clone(&worker.outstanding);
        worker.schedule(0, 8, 0, 16, None).unwrap();
        assert!(worker.take(|| true).is_err());
        drop(worker);
        assert_eq!(reserved.load(Ordering::Acquire), 0);
        assert_eq!(outstanding.load(Ordering::Acquire), 0);
    }
    #[test]
    fn expired_deadline_and_source_mutation_never_publish_bytes() {
        let f = Fixture::new();
        let worker = f.start(8, 1);
        worker.schedule(0, 8, 0, 16, Some(Instant::now())).unwrap();
        assert!(worker.take(|| false).is_err());
        assert_eq!(worker.reserved_bytes(), 0);
        drop(worker);
        // Deterministically order the source mutation before the worker's read so
        // the rejection proves mutation detection, not a lucky read/write race.
        let f = Fixture::new();
        let entered = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        let worker = LocalLookahead::start_with_test_hook(
            f.reader(),
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 8,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
            Box::new(move || {
                worker_entered.wait();
                worker_release.wait();
                Ok(())
            }),
        )
        .unwrap();
        worker.schedule(0, 8, 0, 16, None).unwrap();
        entered.wait(); // worker is parked at before-read, prior to any OS read
        fs::write(&f.0, b"replacement contents different").unwrap();
        release.wait(); // the read now observes the mutated source, never a race
        assert_eq!(
            worker.take(|| false).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(worker.reserved_bytes(), 0);
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn authenticated_ranges_match_source_and_reject_in_place_mutation() {
        use std::io::{Seek, SeekFrom};
        let f = Fixture::new();
        let digest = format!("{:x}", Sha256::digest(fs::read(&f.0).unwrap()));
        let mut reader = f.reader();
        reader
            .authenticate_model(&digest, 8, 128 * 1024, || false)
            .unwrap();
        let worker = LocalLookahead::start(
            reader,
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 8,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
        )
        .unwrap();
        worker.schedule(8, 8, 8, 16, None).unwrap();
        assert_eq!(worker.take(|| false).unwrap().bytes(), b"ijklmnop");
        drop(worker);
        let mut reader = f.reader();
        reader
            .authenticate_model(&digest, 8, 128 * 1024, || false)
            .unwrap();
        let mut writer = fs::OpenOptions::new().write(true).open(&f.0).unwrap();
        writer.seek(SeekFrom::Start(8)).unwrap();
        writer.write_all(b"XXXXXXXX").unwrap();
        let worker = LocalLookahead::start(
            reader,
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 8,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
        )
        .unwrap();
        worker.schedule(8, 8, 8, 16, None).unwrap();
        assert_eq!(
            worker.take(|| false).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(worker.reserved_bytes(), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn consumption_time_authentication_rejects_mutation_after_prefetch() {
        use crate::read_cache::CachedLocalReader;
        use std::io::{Seek, SeekFrom};
        let f = Fixture::new();
        let digest = format!("{:x}", Sha256::digest(fs::read(&f.0).unwrap()));
        let mut consumer = BoundedLocalReader::open(&f.0, 8, 16).unwrap();
        consumer
            .authenticate_model(&digest, 8, 128 * 1024, || false)
            .unwrap();
        let mut consumer = CachedLocalReader::new(consumer, 0).unwrap();

        let mut producer = f.reader();
        producer
            .authenticate_model(&digest, 8, 128 * 1024, || false)
            .unwrap();
        let worker = LocalLookahead::start(
            producer,
            LookaheadLimits {
                staging_bytes: TEST_FIXED_OVERHEAD + 8,
                max_outstanding_reads: 1,
                fixed_overhead_bytes: TEST_FIXED_OVERHEAD,
                worker_stack_bytes: TEST_WORKER_STACK,
            },
        )
        .unwrap();
        worker.schedule(8, 8, 8, 16, None).unwrap();
        let staged = worker.take_exact(8, 8, || false).unwrap();

        let mut writer = fs::OpenOptions::new().write(true).open(&f.0).unwrap();
        writer.seek(SeekFrom::Start(8)).unwrap();
        writer.write_all(b"XXXXXXXX").unwrap();
        assert_eq!(
            consumer
                .verify_staged_range_cancellable(8, staged.bytes(), || false)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        drop(staged);
        assert_eq!(worker.reserved_bytes(), 0);
    }
}
