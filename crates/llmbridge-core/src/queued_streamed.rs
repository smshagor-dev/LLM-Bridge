//! FIFO cooperative admission connected to actual synchronous streamed inference.
//! Reservations account selected budgets; they do not enforce process RSS.
use crate::{
    admission_queue::AdmissionQueue,
    read_cache::cache_allocated_bytes,
    streamed::{generate_streamed_controlled, StreamedConfig, StreamedControl, StreamedMetrics},
};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub struct QueuedStreamedRequest<'a> {
    pub model: &'a Path,
    pub prompt: &'a str,
    pub config: StreamedConfig,
    pub control: StreamedControl<'a>,
    /// Caller-selected additional metadata/runtime allowance, not a measured cap.
    pub additional_reserve_bytes: u64,
}
impl<'a> QueuedStreamedRequest<'a> {
    pub fn new(model: &'a Path, prompt: &'a str) -> Self {
        Self {
            model,
            prompt,
            config: StreamedConfig::default(),
            control: StreamedControl::default(),
            additional_reserve_bytes: 32 * 1024 * 1024,
        }
    }
    /// Reserve full native workspace ceiling, source chunk, exact cache allocation,
    /// optional authentication ceiling and the explicit extra allowance.
    /// This performs no model I/O; native generation still validates configuration.
    pub fn reservation_bytes(&self) -> Result<u64, String> {
        if self.config.workspace_bytes == 0
            || self.config.workspace_bytes > 1024 * 1024 * 1024
            || !(34..=1024 * 1024).contains(&self.config.chunk_bytes)
            || self.config.authentication_budget_bytes == 0
            || self.config.authentication_budget_bytes > 64 * 1024 * 1024
        {
            return Err("invalid queued streamed memory configuration".into());
        }
        let cache = cache_allocated_bytes(self.config.chunk_bytes, self.config.cache_bytes)
            .map_err(|e| e.to_string())? as u64;
        let authentication = if self.control.expected_sha256.is_some() {
            self.config.authentication_budget_bytes as u64
        } else {
            0
        };
        [
            self.config.chunk_bytes as u64,
            cache,
            authentication,
            self.additional_reserve_bytes,
        ]
        .into_iter()
        .try_fold(self.config.workspace_bytes, |sum, bytes| {
            sum.checked_add(bytes)
        })
        .ok_or_else(|| "queued streamed reservation overflow".into())
    }
}
#[derive(Debug)]
pub struct QueuedStreamedMetrics {
    pub inference: StreamedMetrics,
    pub reservation_bytes: u64,
    pub queue_wait_ns: u64,
}

/// Wait cooperatively without opening the model, then hold one reservation across
/// all authentication/native/consumer work. Cancellation, errors and unwinding
/// remove a pending ticket or release the running reservation through RAII.
/// A caller must not synchronously wait on a nested request that needs its own
/// held reservation. Use separate workers/cancellation to coordinate sessions.
pub fn generate_streamed_queued<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    queue: &AdmissionQueue,
    request: QueuedStreamedRequest<'_>,
    cancelled: &AtomicBool,
    on_piece: F,
    on_logits: L,
) -> Result<QueuedStreamedMetrics, String> {
    generate_streamed_queued_with_poll(queue, request, cancelled, None, on_piece, on_logits, || {})
}

/// Bound cooperative admission waiting, not native execution. Zero refuses
/// before enqueue/sampling. Expiration removes a pending ticket through RAII.
pub fn generate_streamed_queued_with_timeout<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    queue: &AdmissionQueue,
    request: QueuedStreamedRequest<'_>,
    cancelled: &AtomicBool,
    wait_timeout: Duration,
    on_piece: F,
    on_logits: L,
) -> Result<QueuedStreamedMetrics, String> {
    generate_streamed_queued_with_poll(
        queue,
        request,
        cancelled,
        Some(wait_timeout),
        on_piece,
        on_logits,
        || {},
    )
}

fn remaining_wait(elapsed: Duration, timeout: Option<Duration>) -> Result<Duration, String> {
    let poll = Duration::from_millis(5);
    match timeout {
        Some(limit) if elapsed >= limit => Err("queued generation admission wait timed out".into()),
        Some(limit) => Ok(poll.min(limit - elapsed)),
        None => Ok(poll),
    }
}

pub(crate) fn generate_streamed_queued_with_poll<
    F: FnMut(&[u8]) -> bool,
    L: FnMut(&[f32]) -> bool,
    P: FnMut(),
>(
    queue: &AdmissionQueue,
    request: QueuedStreamedRequest<'_>,
    cancelled: &AtomicBool,
    wait_timeout: Option<Duration>,
    on_piece: F,
    on_logits: L,
    mut before_poll: P,
) -> Result<QueuedStreamedMetrics, String> {
    if cancelled.load(Ordering::Acquire) {
        return Err("queued generation cancelled before admission".into());
    }
    crate::native_cpu::ensure_compatible()?;
    let charge = request.reservation_bytes()?;
    let start = Instant::now();
    remaining_wait(start.elapsed(), wait_timeout)?;
    before_poll();
    remaining_wait(start.elapsed(), wait_timeout)?;
    let mut pending = queue
        .request(charge)
        .map_err(|e| format!("streamed queue admission: {e:?}"))?;
    let reservation = loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("queued generation cancelled while waiting".into());
        }
        remaining_wait(start.elapsed(), wait_timeout)?;
        before_poll();
        remaining_wait(start.elapsed(), wait_timeout)?;
        if let Some(reservation) = pending
            .try_admit()
            .map_err(|e| format!("streamed queue polling: {e:?}"))?
        {
            break reservation;
        }
        std::thread::park_timeout(remaining_wait(start.elapsed(), wait_timeout)?);
    };
    let queue_wait_ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    if cancelled.load(Ordering::Acquire) {
        return Err("queued generation cancelled after admission".into());
    }
    remaining_wait(start.elapsed(), wait_timeout)?;
    let inference = generate_streamed_controlled(
        request.model,
        request.prompt,
        request.config,
        cancelled,
        request.control,
        on_piece,
        on_logits,
    );
    drop(reservation);
    Ok(QueuedStreamedMetrics {
        inference: inference?,
        reservation_bytes: charge,
        queue_wait_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn waiting_boundaries_do_not_overflow_or_oversleep_remaining_budget() {
        let limit = Duration::from_millis(10);
        assert!(remaining_wait(Duration::ZERO, Some(Duration::ZERO)).is_err());
        assert!(remaining_wait(limit, Some(limit)).is_err());
        assert!(remaining_wait(limit + limit, Some(limit)).is_err());
        assert_eq!(
            remaining_wait(Duration::from_millis(9), Some(limit)).unwrap(),
            Duration::from_millis(1)
        );
        assert_eq!(
            remaining_wait(Duration::ZERO, Some(Duration::MAX)).unwrap(),
            Duration::from_millis(5)
        );
        assert_eq!(
            remaining_wait(Duration::MAX, None).unwrap(),
            Duration::from_millis(5)
        );
    }
    #[test]
    fn expired_sampler_and_zero_budget_do_not_open_model_or_leak_ticket() {
        use crate::memory_budget::MemoryBudget;
        let budget = MemoryBudget::new(64 * 1024 * 1024).unwrap();
        let queue = AdmissionQueue::new(budget.clone(), 1).unwrap();
        let request = QueuedStreamedRequest::new(Path::new("not-opened.gguf"), "fixture");
        let error = generate_streamed_queued_with_poll(
            &queue,
            request,
            &AtomicBool::new(false),
            Some(Duration::ZERO),
            |_| panic!("output"),
            |_| panic!("logits"),
            || panic!("zero timeout sampled"),
        )
        .unwrap_err();
        assert!(error.contains("admission wait timed out"));
        let mut sampled_pending = false;
        let error = generate_streamed_queued_with_poll(
            &queue,
            request,
            &AtomicBool::new(false),
            Some(Duration::from_millis(10)),
            |_| panic!("output"),
            |_| panic!("logits"),
            || {
                if queue.pending() == 1 {
                    sampled_pending = true;
                    std::thread::sleep(Duration::from_millis(20));
                }
            },
        )
        .unwrap_err();
        assert!(sampled_pending);
        assert!(error.contains("admission wait timed out"));
        assert_eq!(queue.pending(), 0);
        assert_eq!(budget.reserved(), 0);
        let error = generate_streamed_queued_with_timeout(
            &queue,
            request,
            &AtomicBool::new(true),
            Duration::ZERO,
            |_| panic!("output"),
            |_| panic!("logits"),
        )
        .unwrap_err();
        assert!(error.contains("cancelled before admission"));
    }
    #[test]
    fn charge_includes_all_selected_ceilings_and_exact_cache_slots() {
        let mut request = QueuedStreamedRequest::new(Path::new("not-opened.gguf"), "fixture");
        request.additional_reserve_bytes = 100;
        request.config.cache_bytes = 1024 * 1024;
        let expected = request.config.workspace_bytes + 65536 + 983376 + 100;
        assert_eq!(request.reservation_bytes().unwrap(), expected);
        request.control.expected_sha256 = Some("trusted");
        assert_eq!(
            request.reservation_bytes().unwrap(),
            expected + 8 * 1024 * 1024
        );
        request.additional_reserve_bytes = u64::MAX;
        assert!(request.reservation_bytes().is_err());
        request.config.chunk_bytes = 0;
        assert!(request.reservation_bytes().is_err());
    }
}
