//! Cooperative admission hysteresis, independent of native allocation/RSS limits.
#[cfg(any(feature = "native", test))]
use crate::admission_queue::AdmissionQueue;
use std::time::Duration;
#[cfg(any(feature = "native", test))]
use std::{fmt, sync::Mutex, time::Instant};

#[derive(Debug, Clone, Copy)]
pub struct MemoryPressurePolicy {
    /// Pause when effective available RAM is strictly below this floor.
    pub pause_below_bytes: u64,
    /// Recover only at or above this higher threshold.
    pub resume_at_bytes: u64,
    /// Shared minimum sample interval; queued callers still check cancellation.
    pub sample_interval: Duration,
}
impl MemoryPressurePolicy {
    #[cfg(any(feature = "native", test))]
    pub(crate) fn validate(&self, minimum_floor: u64) -> Result<(), &'static str> {
        if self.pause_below_bytes < minimum_floor || self.pause_below_bytes == 0 {
            return Err("pressure floor must cover the OS reserve and one full session slot");
        }
        if self.resume_at_bytes <= self.pause_below_bytes {
            return Err("pressure recovery threshold must exceed pause floor");
        }
        if self.sample_interval.is_zero() || self.sample_interval > Duration::from_secs(60) {
            return Err("pressure sample interval must be positive and at most 60 seconds");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PressureSnapshot {
    pub paused: bool,
    pub last_available_bytes: Option<u64>,
    pub sample_count: u64,
}
#[cfg(any(feature = "native", test))]
struct State {
    snapshot: PressureSnapshot,
    last_sample: Option<Instant>,
}
#[cfg(any(feature = "native", test))]
pub(crate) struct PressureGate {
    policy: MemoryPressurePolicy,
    state: Mutex<State>,
    sampler: Box<dyn Fn() -> Option<u64> + Send + Sync>,
}
#[cfg(any(feature = "native", test))]
impl fmt::Debug for PressureGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PressureGate")
            .field("policy", &self.policy)
            .field("snapshot", &self.snapshot())
            .finish_non_exhaustive()
    }
}
#[cfg(any(feature = "native", test))]
impl PressureGate {
    pub(crate) fn new(
        policy: MemoryPressurePolicy,
        sampler: impl Fn() -> Option<u64> + Send + Sync + 'static,
    ) -> Self {
        Self {
            policy,
            state: Mutex::new(State {
                snapshot: PressureSnapshot {
                    paused: true,
                    last_available_bytes: None,
                    sample_count: 0,
                },
                last_sample: None,
            }),
            sampler: Box::new(sampler),
        }
    }
    pub(crate) fn snapshot(&self) -> PressureSnapshot {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot
    }
    pub(crate) fn refresh(&self, queue: &AdmissionQueue) {
        // Serialize samples and gate application so an older observation cannot
        // overwrite a newer decision. No queue/budget lock is held during I/O.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state
            .last_sample
            .is_some_and(|last| last.elapsed() < self.policy.sample_interval)
        {
            return;
        }
        let available = (self.sampler)();
        let paused = match available {
            None => true,
            Some(bytes) if state.snapshot.paused => bytes < self.policy.resume_at_bytes,
            Some(bytes) => bytes < self.policy.pause_below_bytes,
        };
        state.snapshot = PressureSnapshot {
            paused,
            last_available_bytes: available,
            sample_count: state.snapshot.sample_count.saturating_add(1),
        };
        queue.set_policy_paused(paused);
        state.last_sample = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_budget::MemoryBudget;
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    };
    fn policy() -> MemoryPressurePolicy {
        MemoryPressurePolicy {
            pause_below_bytes: 100,
            resume_at_bytes: 200,
            sample_interval: Duration::from_nanos(1),
        }
    }
    #[test]
    fn thresholds_and_intervals_fail_closed() {
        assert!(policy().validate(100).is_ok());
        assert!(policy().validate(101).is_err());
        for p in [
            MemoryPressurePolicy {
                pause_below_bytes: 0,
                ..policy()
            },
            MemoryPressurePolicy {
                resume_at_bytes: 100,
                ..policy()
            },
            MemoryPressurePolicy {
                sample_interval: Duration::ZERO,
                ..policy()
            },
            MemoryPressurePolicy {
                sample_interval: Duration::from_secs(61),
                ..policy()
            },
        ] {
            assert!(p.validate(0).is_err());
        }
    }
    #[test]
    fn hysteresis_unknown_and_manual_pause_keep_gates_independent() {
        let value = Arc::new(AtomicU64::new(150));
        let sample = Arc::clone(&value);
        let gate = PressureGate::new(policy(), move || match sample.load(Ordering::Acquire) {
            u64::MAX => None,
            n => Some(n),
        });
        let budget = MemoryBudget::new(100).unwrap();
        let queue = AdmissionQueue::new(Arc::clone(&budget), 4).unwrap();
        let mut pending = queue.request(100).unwrap();
        let refresh = |n| {
            value.store(n, Ordering::Release);
            // The gate re-samples only when the sample_interval (1ns here) has
            // elapsed since the last sample. On coarse-granularity monotonic
            // clocks (notably Windows `Instant`) two rapid calls can read the
            // same instant (0ns elapsed), so the throttle would skip sampling and
            // reuse a stale value. Spin until the clock advances past the interval
            // so every refresh deterministically re-samples. This changes no
            // assertion; it only removes a cross-platform timing flake.
            let t = Instant::now();
            while t.elapsed() < Duration::from_nanos(1) {
                std::hint::spin_loop();
            }
            gate.refresh(&queue);
            gate.snapshot().paused
        };
        assert!(refresh(150)); // Start closed until recovery threshold.
        assert!(pending.try_admit().unwrap().is_none());
        assert!(!refresh(200));
        assert!(!refresh(100)); // Exact low boundary remains open.
        let held = pending.try_admit().unwrap().unwrap();
        assert!(refresh(99));
        assert_eq!(budget.reserved(), 100); // Pressure never revokes a charge.
        let mut tail = queue.request(100).unwrap();
        drop(held);
        assert!(tail.try_admit().unwrap().is_none());
        assert!(refresh(199));
        queue.set_paused(true);
        assert!(!refresh(200));
        assert!(queue.is_paused());
        assert!(tail.try_admit().unwrap().is_none());
        queue.set_paused(false);
        assert!(refresh(u64::MAX)); // Unknown closes an open gate.
        assert!(!queue.is_paused());
        assert!(tail.try_admit().unwrap().is_none());
        assert!(!refresh(200));
        assert!(tail.try_admit().unwrap().is_some());
        assert_eq!(budget.reserved(), 0);
    }
    #[test]
    fn sampling_interval_is_shared_and_not_per_poll() {
        let gate = PressureGate::new(
            MemoryPressurePolicy {
                sample_interval: Duration::from_secs(60),
                ..policy()
            },
            || Some(200),
        );
        let queue = AdmissionQueue::new(MemoryBudget::new(100).unwrap(), 4).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..100 {
                        gate.refresh(&queue);
                    }
                });
            }
        });
        assert_eq!(gate.snapshot().sample_count, 1);
    }
}
