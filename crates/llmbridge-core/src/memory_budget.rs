//! Atomic cooperative memory reservations shared across concurrent inference requests.
//! This bounds *accounted reservations*, not native allocations or process RSS.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub struct MemoryBudget {
    capacity: u64,
    reserved: AtomicU64,
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
            capacity,
            reserved: AtomicU64::new(0),
        }))
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn reserved(&self) -> u64 {
        self.reserved.load(Ordering::Acquire)
    }

    /// All concurrent callers compete for one limit. No optimistic overcommit.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<Reservation, &'static str> {
        if bytes == 0 {
            return Err("reservation must be positive");
        }
        let mut current = self.reserved.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(bytes).ok_or("reservation overflow")?;
            if next > self.capacity {
                return Err("insufficient unreserved budget");
            }
            match self.reserved.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Reservation {
                        budget: Arc::clone(self),
                        bytes,
                    })
                }
                Err(observed) => current = observed,
            }
        }
    }
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let previous = self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes, "reservation accounting underflow");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
