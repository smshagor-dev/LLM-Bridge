//! Cooperative FIFO admission for concurrent CPU inference sessions.
//! This does not enforce native allocations or process RSS. Callers must hold
//! the returned reservation for the entire execution lifetime.
use crate::memory_budget::{MemoryBudget, Reservation};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidRequest,
    QueueFull,
    TicketExhausted,
    UnknownTicket,
    InsufficientCapacity,
}

#[derive(Debug)]
struct Pending {
    ticket: Ticket,
    bytes: u64,
}

#[derive(Debug, Default)]
struct QueueState {
    next: u64,
    pending: VecDeque<Pending>,
}

/// FIFO admission deliberately does not skip an oversized head request.
/// Cancelling a ticket is explicit; dropping a ticket value does not cancel it.
#[derive(Debug)]
pub struct AdmissionQueue {
    budget: Arc<MemoryBudget>,
    max_pending: usize,
    state: Mutex<QueueState>,
}

impl AdmissionQueue {
    pub fn new(budget: Arc<MemoryBudget>, max_pending: usize) -> Result<Self, AdmissionError> {
        if max_pending == 0 {
            return Err(AdmissionError::InvalidRequest);
        }
        Ok(Self {
            budget,
            max_pending,
            state: Mutex::new(QueueState::default()),
        })
    }

    /// Enqueue without reserving any memory. A request larger than the entire
    /// capacity is refused immediately instead of blocking the queue forever.
    pub fn enqueue(&self, bytes: u64) -> Result<Ticket, AdmissionError> {
        if bytes == 0 {
            return Err(AdmissionError::InvalidRequest);
        }
        if bytes > self.budget.capacity() {
            return Err(AdmissionError::InsufficientCapacity);
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.pending.len() >= self.max_pending {
            return Err(AdmissionError::QueueFull);
        }
        let next = state.next.checked_add(1).ok_or(AdmissionError::TicketExhausted)?;
        state.next = next;
        let ticket = Ticket(next);
        state.pending.push_back(Pending { ticket, bytes });
        Ok(ticket)
    }

    /// Nonblocking poll. Only the oldest live ticket can acquire a reservation.
    /// None means wait/retry; an unknown or cancelled ticket is an error.
    /// Reservation acquisition and head removal occur under the same lock.
    pub fn try_admit(&self, ticket: Ticket) -> Result<Option<Reservation>, AdmissionError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.pending.front().is_some_and(|p| p.ticket == ticket) {
            let bytes = state.pending.front().expect("front checked").bytes;
            match self.budget.try_reserve(bytes) {
                Ok(reservation) => {
                    state.pending.pop_front();
                    Ok(Some(reservation))
                }
                Err(_) => Ok(None),
            }
        } else if state.pending.iter().any(|p| p.ticket == ticket) {
            Ok(None)
        } else {
            Err(AdmissionError::UnknownTicket)
        }
    }

    /// Idempotent cancellation removes a waiting ticket, never a running
    /// reservation. Running reservations are released only when dropped.
    pub fn cancel(&self, ticket: Ticket) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(position) = state.pending.iter().position(|p| p.ticket == ticket) {
            state.pending.remove(position);
            true
        } else {
            false
        }
    }

    pub fn pending(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_waits_for_head_even_when_tail_fits() {
        let budget = MemoryBudget::new(100).unwrap();
        let held = budget.try_reserve(60).unwrap();
        let queue = AdmissionQueue::new(Arc::clone(&budget), 4).unwrap();
        let first = queue.enqueue(70).unwrap();
        let second = queue.enqueue(20).unwrap();
        assert!(queue.try_admit(second).unwrap().is_none());
        assert!(queue.try_admit(first).unwrap().is_none());
        drop(held);
        let first_reservation = queue.try_admit(first).unwrap().unwrap();
        let second_reservation = queue.try_admit(second).unwrap().unwrap();
        assert_eq!(budget.reserved(), 90);
        drop(first_reservation);
        drop(second_reservation);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn cancellation_unblocks_tail_without_leaking_budget() {
        let budget = MemoryBudget::new(100).unwrap();
        let queue = AdmissionQueue::new(Arc::clone(&budget), 2).unwrap();
        let head = queue.enqueue(90).unwrap();
        let tail = queue.enqueue(20).unwrap();
        assert!(queue.cancel(head));
        assert!(!queue.cancel(head));
        assert_eq!(queue.try_admit(head).unwrap_err(), AdmissionError::UnknownTicket);
        let reservation = queue.try_admit(tail).unwrap().unwrap();
        assert_eq!(queue.pending(), 0);
        assert!(!queue.cancel(tail));
        drop(reservation);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn rejects_invalid_and_unserviceable_requests() {
        let budget = MemoryBudget::new(10).unwrap();
        assert_eq!(AdmissionQueue::new(Arc::clone(&budget), 0).unwrap_err(), AdmissionError::InvalidRequest);
        let queue = AdmissionQueue::new(budget, 1).unwrap();
        assert_eq!(queue.enqueue(0).unwrap_err(), AdmissionError::InvalidRequest);
        assert_eq!(queue.enqueue(11).unwrap_err(), AdmissionError::InsufficientCapacity);
        let ticket = queue.enqueue(10).unwrap();
        assert_eq!(queue.enqueue(1).unwrap_err(), AdmissionError::QueueFull);
        assert!(queue.cancel(ticket));
        assert_eq!(queue.pending(), 0);
    }

    #[test]
    fn competing_polls_never_duplicate_a_reservation() {
        let budget = MemoryBudget::new(10).unwrap();
        let queue = Arc::new(AdmissionQueue::new(Arc::clone(&budget), 1).unwrap());
        let ticket = queue.enqueue(10).unwrap();
        let gate = Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let queue = Arc::clone(&queue);
            let gate = Arc::clone(&gate);
            workers.push(std::thread::spawn(move || {
                gate.wait();
                queue.try_admit(ticket).ok().flatten()
            }));
        }
        gate.wait();
        let admitted: Vec<_> = workers.into_iter().filter_map(|w| w.join().unwrap()).collect();
        assert_eq!(admitted.len(), 1);
        assert_eq!(budget.reserved(), 10);
        drop(admitted);
        assert_eq!(budget.reserved(), 0);
    }
}
