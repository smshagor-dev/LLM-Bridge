//! Fixed-slot admission connects CPU/RAM worker planning to real streamed sessions.
//! Callers own threads; one shared executor bounds admitted sessions, not OS RSS.
use crate::{
    admission_queue::AdmissionQueue,
    hardware::Hardware,
    memory_budget::MemoryBudget,
    queued_streamed::{generate_streamed_queued, QueuedStreamedMetrics, QueuedStreamedRequest},
    worker_plan::{plan_workers, WorkerPlan, WorkerRequest},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Debug, Clone, Copy)]
pub struct ExecutorConfig {
    pub requested_workers: usize,
    /// Full accounted slot ceiling for every admitted session, including its allowances.
    pub session_slot_bytes: u64,
    pub os_reserve_bytes: u64,
    pub max_pending: usize,
}
impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            requested_workers: 1,
            session_slot_bytes: 64 * 1024 * 1024,
            os_reserve_bytes: 512 * 1024 * 1024,
            max_pending: 32,
        }
    }
}
#[derive(Debug)]
pub struct StreamedExecutor {
    queue: AdmissionQueue,
    budget: Arc<MemoryBudget>,
    plan: WorkerPlan,
    slot_bytes: u64,
}
impl StreamedExecutor {
    /// Uses the caller's memory/CPU snapshot; no model I/O or native execution.
    /// Share the resulting instance across competing caller threads.
    pub fn new(hardware: &Hardware, config: ExecutorConfig) -> Result<Self, String> {
        if config.session_slot_bytes == 0 || config.max_pending == 0 {
            return Err("executor slot and pending capacity must be positive".into());
        }
        let plan = plan_workers(
            hardware,
            WorkerRequest {
                shared_bytes: 0,
                scratch_per_worker_bytes: config.session_slot_bytes,
                already_reserved_bytes: 0,
                reserve_bytes: config.os_reserve_bytes,
                requested_workers: config.requested_workers,
            },
        )?;
        let capacity = (plan.workers as u64)
            .checked_mul(config.session_slot_bytes)
            .ok_or("executor capacity overflow")?;
        let budget = MemoryBudget::new(capacity).map_err(|e| e.to_string())?;
        let queue = AdmissionQueue::new(Arc::clone(&budget), config.max_pending)
            .map_err(|e| format!("executor queue: {e:?}"))?;
        Ok(Self {
            queue,
            budget,
            plan,
            slot_bytes: config.session_slot_bytes,
        })
    }
    pub fn worker_count(&self) -> usize {
        self.plan.workers
    }
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }
    pub fn capacity_bytes(&self) -> u64 {
        self.budget.capacity()
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.budget.reserved()
    }
    pub fn active_sessions(&self) -> u64 {
        self.budget.reserved() / self.slot_bytes
    }
    pub fn pending_sessions(&self) -> usize {
        self.queue.pending()
    }

    /// Reserves a full fixed slot even for a smaller request. Charging only its
    /// actual ceiling could otherwise admit more sessions than planned CPU workers.
    /// Existing controlled generation performs all native/config/source checks.
    pub fn generate<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
        &self,
        mut request: QueuedStreamedRequest<'_>,
        cancelled: &AtomicBool,
        on_piece: F,
        on_logits: L,
    ) -> Result<QueuedStreamedMetrics, String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("executor session cancelled before admission".into());
        }
        crate::native_cpu::ensure_compatible()?;
        let requested = request.reservation_bytes()?;
        if requested > self.slot_bytes {
            return Err(format!(
                "session reservation {requested} exceeds executor slot {}",
                self.slot_bytes
            ));
        }
        request.additional_reserve_bytes = request
            .additional_reserve_bytes
            .checked_add(self.slot_bytes - requested)
            .ok_or("executor allowance overflow")?;
        generate_streamed_queued(&self.queue, request, cancelled, on_piece, on_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hardware(cpus: usize, available: Option<u64>) -> Hardware {
        Hardware {
            logical_cpus: cpus,
            total_ram_bytes: available,
            available_ram_bytes: available,
        }
    }
    fn config() -> ExecutorConfig {
        ExecutorConfig {
            requested_workers: 8,
            session_slot_bytes: 100,
            os_reserve_bytes: 50,
            max_pending: 4,
        }
    }
    #[test]
    fn worker_plan_bounds_capacity_by_cpu_memory_and_request() {
        let memory = StreamedExecutor::new(&hardware(8, Some(350)), config()).unwrap();
        assert_eq!((memory.worker_count(), memory.capacity_bytes()), (3, 300));
        let cpu = StreamedExecutor::new(&hardware(2, Some(1000)), config()).unwrap();
        assert_eq!((cpu.worker_count(), cpu.capacity_bytes()), (2, 200));
        let requested = StreamedExecutor::new(
            &hardware(8, Some(1000)),
            ExecutorConfig {
                requested_workers: 1,
                ..config()
            },
        )
        .unwrap();
        assert_eq!(requested.capacity_bytes(), 100);
        assert_eq!(requested.active_sessions(), 0);
        assert_eq!(requested.pending_sessions(), 0);
    }
    #[test]
    fn unknown_exhausted_or_invalid_plans_do_not_create_executors() {
        assert!(StreamedExecutor::new(&hardware(8, None), config()).is_err());
        assert!(StreamedExecutor::new(&hardware(8, Some(149)), config()).is_err());
        for c in [
            ExecutorConfig {
                requested_workers: 0,
                ..config()
            },
            ExecutorConfig {
                session_slot_bytes: 0,
                ..config()
            },
            ExecutorConfig {
                max_pending: 0,
                ..config()
            },
            ExecutorConfig {
                os_reserve_bytes: u64::MAX,
                ..config()
            },
        ] {
            assert!(StreamedExecutor::new(&hardware(8, Some(1000)), c).is_err());
        }
    }
}
