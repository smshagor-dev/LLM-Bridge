//! Constant-space, opt-in streamed-session observations. These are wall times,
//! not a physical-storage benchmark or a CPU-only compute measurement.
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default)]
pub struct SessionPerformance {
    pub total_ns: u64,
    pub first_logits_ready_ns: Option<u64>,
    pub first_piece_ns: Option<u64>,
    pub sampling_steps: u64,
    pub emitted_pieces: u64,
    pub emitted_bytes: u64,
    pub decode_intervals: u64,
    pub decode_elapsed_ns: u64,
    pub range_callback_ns: u64,
    pub consumer_callback_ns: u64,
}
impl SessionPerformance {
    pub fn decode_steps_per_second(&self) -> Option<f64> {
        (self.decode_intervals > 0 && self.decode_elapsed_ns > 0)
            .then(|| self.decode_intervals as f64 * 1e9 / self.decode_elapsed_ns as f64)
    }
    pub fn to_json(&self) -> String {
        let integer = |v: Option<u64>| v.map_or_else(|| "null".into(), |n| n.to_string());
        let rate = self
            .decode_steps_per_second()
            .map_or_else(|| "null".into(), |n| n.to_string());
        format!(concat!("{{\"total_ns\":{},\"first_logits_ready_ns\":{},\"first_piece_ns\":{},",
            "\"sampling_steps\":{},\"emitted_pieces\":{},\"emitted_bytes\":{},",
            "\"decode_intervals\":{},\"decode_elapsed_ns\":{},\"decode_steps_per_second\":{},",
            "\"range_callback_ns\":{},\"consumer_callback_ns\":{},",
            "\"includes_admission_and_authentication\":true,\"decode_includes_consumer_callbacks\":true,",
            "\"physical_storage_time_measured\":false}}"),
            self.total_ns, integer(self.first_logits_ready_ns), integer(self.first_piece_ns),
            self.sampling_steps, self.emitted_pieces, self.emitted_bytes, self.decode_intervals,
            self.decode_elapsed_ns, rate, self.range_callback_ns, self.consumer_callback_ns)
    }
    fn logits_at(&mut self, elapsed: u64) {
        let first = *self.first_logits_ready_ns.get_or_insert(elapsed);
        self.sampling_steps = self.sampling_steps.saturating_add(1);
        self.decode_intervals = self.sampling_steps - 1;
        self.decode_elapsed_ns = elapsed.saturating_sub(first);
    }
    fn piece_at(&mut self, elapsed: u64, bytes: usize) {
        // Native emits one callback per non-EOG token, including empty pieces.
        // An empty piece is not visible first output.
        if bytes > 0 {
            self.first_piece_ns.get_or_insert(elapsed);
        }
        self.emitted_pieces = self.emitted_pieces.saturating_add(1);
        self.emitted_bytes = self.emitted_bytes.saturating_add(bytes as u64);
    }
}

pub(crate) struct PerformanceClock {
    start: Instant,
    values: SessionPerformance,
}
fn nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}
impl PerformanceClock {
    pub(crate) fn start() -> Self {
        Self {
            start: Instant::now(),
            values: SessionPerformance::default(),
        }
    }
    pub(crate) fn logits(&mut self) {
        self.values.logits_at(nanos(self.start.elapsed()));
    }
    pub(crate) fn piece(&mut self, bytes: usize) {
        self.values.piece_at(nanos(self.start.elapsed()), bytes);
    }
    pub(crate) fn range(&mut self, duration: Duration) {
        self.values.range_callback_ns = self
            .values
            .range_callback_ns
            .saturating_add(nanos(duration));
    }
    pub(crate) fn consumer(&mut self, duration: Duration) {
        self.values.consumer_callback_ns = self
            .values
            .consumer_callback_ns
            .saturating_add(nanos(duration));
    }
    pub(crate) fn finish(mut self) -> SessionPerformance {
        self.values.total_ns = nanos(self.start.elapsed());
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_rate_excludes_prefill_and_has_n_minus_one_intervals() {
        let mut p = SessionPerformance::default();
        p.logits_at(9_000_000_000);
        assert_eq!(p.decode_steps_per_second(), None);
        p.logits_at(9_500_000_000);
        p.logits_at(10_000_000_000);
        assert_eq!(p.decode_intervals, 2);
        assert_eq!(p.decode_steps_per_second(), Some(2.0));
        assert_eq!(p.first_logits_ready_ns, Some(9_000_000_000));
    }
    #[test]
    fn eos_sampling_and_empty_pieces_are_not_visible_tokens() {
        let mut p = SessionPerformance::default();
        p.logits_at(10);
        p.piece_at(12, 0);
        assert_eq!(p.first_piece_ns, None);
        p.logits_at(20);
        p.piece_at(22, 3);
        p.logits_at(30); // EOG: no emit callback.
        assert_eq!(p.sampling_steps, 3);
        assert_eq!(p.emitted_pieces, 2);
        assert_eq!(p.emitted_bytes, 3);
        assert_eq!(p.first_piece_ns, Some(22));
    }
    #[test]
    fn zero_duration_and_no_output_are_unknown_not_infinite() {
        let mut p = SessionPerformance::default();
        assert!(p.to_json().contains("\"decode_steps_per_second\":null"));
        p.logits_at(1);
        p.logits_at(1);
        assert_eq!(p.decode_steps_per_second(), None);
    }
    #[test]
    fn duration_and_byte_counters_saturate() {
        assert_eq!(nanos(Duration::MAX), u64::MAX);
        let mut p = SessionPerformance {
            emitted_bytes: u64::MAX,
            ..Default::default()
        };
        p.piece_at(0, 1);
        assert_eq!(p.emitted_bytes, u64::MAX);
    }
}
