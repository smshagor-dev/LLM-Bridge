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
    if cancelled.load(Ordering::Acquire) {
        return Err("queued generation cancelled before admission".into());
    }
    crate::native_cpu::ensure_compatible()?;
    let charge = request.reservation_bytes()?;
    let start = Instant::now();
    let mut pending = queue
        .request(charge)
        .map_err(|e| format!("streamed queue admission: {e:?}"))?;
    let reservation = loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("queued generation cancelled while waiting".into());
        }
        if let Some(reservation) = pending
            .try_admit()
            .map_err(|e| format!("streamed queue polling: {e:?}"))?
        {
            break reservation;
        }
        std::thread::park_timeout(Duration::from_millis(5));
    };
    let queue_wait_ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    if cancelled.load(Ordering::Acquire) {
        return Err("queued generation cancelled after admission".into());
    }
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
