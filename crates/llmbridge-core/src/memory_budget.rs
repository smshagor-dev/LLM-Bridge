//! Synchronized cooperative memory reservations shared across inference requests.
//! New admissions obey current capacity. Shrinking preserves existing charges,
//! which can temporarily exceed capacity; this is not an OS allocation/RSS cap.
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub capacity_bytes: u64,
    pub reserved_bytes: u64,
    pub available_bytes: u64,
    pub over_capacity_bytes: u64,
}
#[derive(Debug)]
struct BudgetState {
    capacity: u64,
    reserved: u64,
}
#[derive(Debug)]
pub struct MemoryBudget {
    state: Mutex<BudgetState>,
}
#[derive(Debug)]
pub struct Reservation {
    budget: Arc<MemoryBudget>,
    bytes: u64,
}

impl MemoryBudget {
    pub fn new(capacity: u64) -> Result<Arc<Self>, &'static str> {
        if capacity == 0 {
            return Err("memory budget must be positive");
        }
        Ok(Arc::new(Self {
            state: Mutex::new(BudgetState {
                capacity,
                reserved: 0,
            }),
        }))
    }
    pub fn capacity(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .capacity
    }
    pub fn reserved(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reserved
    }
    /// Atomic snapshot: the fields always describe one state, including shrink debt.
    pub fn snapshot(&self) -> BudgetSnapshot {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        BudgetSnapshot {
            capacity_bytes: state.capacity,
            reserved_bytes: state.reserved,
            available_bytes: state.capacity.saturating_sub(state.reserved),
            over_capacity_bytes: state.reserved.saturating_sub(state.capacity),
        }
    }
    /// Change admission capacity without revoking existing Reservation values.
    /// Zero blocks new reservations; construction still requires positive capacity.
    /// Returns the previous capacity. Serializes with acquire/release/snapshot.
    pub fn set_capacity(&self, capacity: u64) -> u64 {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::replace(&mut state.capacity, capacity)
    }
    /// Every concurrent acquisition checks and charges the same current capacity
    /// under one lock, so a completed shrink cannot admit from a stale limit.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<Reservation, &'static str> {
        if bytes == 0 {
            return Err("reservation must be positive");
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let next = state
            .reserved
            .checked_add(bytes)
            .ok_or("reservation overflow")?;
        if next > state.capacity {
            return Err("insufficient unreserved budget");
        }
        state.reserved = next;
        Ok(Reservation {
            budget: Arc::clone(self),
            bytes,
        })
    }
}
impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reserved = state
            .reserved
            .checked_sub(self.bytes)
            .expect("reservation accounting underflow");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shrink_preserves_charges_and_reports_debt_until_release() {
        let budget = MemoryBudget::new(100).unwrap();
        let held = budget.try_reserve(80).unwrap();
        assert_eq!(budget.set_capacity(20), 100);
        assert_eq!(
            budget.snapshot(),
            BudgetSnapshot {
                capacity_bytes: 20,
                reserved_bytes: 80,
                available_bytes: 0,
                over_capacity_bytes: 60
            }
        );
        assert!(budget.try_reserve(1).is_err());
        assert_eq!(held.bytes(), 80);
        drop(held);
        assert_eq!(
            budget.snapshot(),
            BudgetSnapshot {
                capacity_bytes: 20,
                reserved_bytes: 0,
                available_bytes: 20,
                over_capacity_bytes: 0
            }
        );
        let exact = budget.try_reserve(20).unwrap();
        assert!(budget.try_reserve(1).is_err());
        drop(exact);
    }
    #[test]
    fn zero_capacity_gates_new_acquisitions_and_growth_reopens_them() {
        let budget = MemoryBudget::new(100).unwrap();
        let held = budget.try_reserve(40).unwrap();
        assert_eq!(budget.set_capacity(0), 100);
        assert_eq!(budget.set_capacity(0), 0);
        assert_eq!(budget.reserved(), 40);
        drop(held);
        assert!(budget.try_reserve(1).is_err());
        assert_eq!(budget.set_capacity(60), 0);
        let held = budget.try_reserve(60).unwrap();
        assert_eq!(budget.snapshot().available_bytes, 0);
        drop(held);
        assert_eq!(budget.snapshot().available_bytes, 60);
    }
    #[test]
    fn concurrent_callers_cannot_acquire_after_completed_zero_resize() {
        use std::sync::Barrier;
        let budget = MemoryBudget::new(100).unwrap();
        let held = budget.try_reserve(60).unwrap();
        budget.set_capacity(0);
        let gate = Arc::new(Barrier::new(9));
        let mut callers = Vec::new();
        for _ in 0..8 {
            let budget = Arc::clone(&budget);
            let gate = Arc::clone(&gate);
            callers.push(std::thread::spawn(move || {
                gate.wait();
                budget.try_reserve(1).ok()
            }));
        }
        gate.wait();
        for caller in callers {
            assert!(caller.join().unwrap().is_none());
        }
        assert_eq!(budget.reserved(), 60);
        drop(held);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        assert_eq!(budget.snapshot().capacity_bytes, 0);
    }

    #[test]
    fn shrink_keeps_fifo_head_until_it_can_fit_or_is_cancelled() {
        let budget = MemoryBudget::new(100).unwrap();
        let queue = crate::admission_queue::AdmissionQueue::new(Arc::clone(&budget), 2).unwrap();
        let head = queue.enqueue(100).unwrap();
        let tail = queue.enqueue(10).unwrap();
        budget.set_capacity(20);
        assert!(queue.try_admit(head).unwrap().is_none());
        assert!(queue.try_admit(tail).unwrap().is_none());
        assert_eq!(
            queue.enqueue(1).unwrap_err(),
            crate::admission_queue::AdmissionError::QueueFull
        );
        assert!(queue.cancel(head));
        let admitted = queue.try_admit(tail).unwrap().unwrap();
        assert_eq!(budget.reserved(), 10);
        assert_eq!(budget.snapshot().available_bytes, 10);
        drop(admitted);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn capacity_is_exclusive_and_drop_releases() {
        let budget = MemoryBudget::new(100).unwrap();
        let first = budget.try_reserve(70).unwrap();
        assert!(budget.try_reserve(31).is_err());
        let second = budget.try_reserve(30).unwrap();
        assert_eq!(budget.reserved(), 100);
        drop(first);
        assert_eq!(budget.reserved(), 30);
        drop(second);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn zero_and_overflow_fail_closed() {
        assert!(MemoryBudget::new(0).is_err());
        let budget = MemoryBudget::new(u64::MAX).unwrap();
        assert!(budget.try_reserve(0).is_err());
        let full = budget.try_reserve(u64::MAX).unwrap();
        assert!(budget.try_reserve(1).is_err());
        drop(full);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn concurrent_reservations_never_overcommit() {
        use std::sync::{Arc, Barrier};
        let budget = MemoryBudget::new(100).unwrap();
        let barrier = Arc::new(Barrier::new(9));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let budget = Arc::clone(&budget);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                budget.try_reserve(60).ok()
            }));
        }
        barrier.wait();
        let reservations: Vec<_> = handles
            .into_iter()
            .filter_map(|h| h.join().unwrap())
            .collect();
        assert_eq!(reservations.len(), 1);
        assert_eq!(budget.reserved(), 60);
        drop(reservations);
        assert_eq!(budget.reserved(), 0);
    }
}
