//! Bounded token work windows for a future batched CPU dispatcher.
//! Planning only: this does not alter native prefill/decode or enforce process RSS.
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenWindow {
    pub start: u64,
    pub end: u64,
    /// Maximum scratch reservation for this window, including fixed overhead.
    pub reserved_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenWindowLimits {
    pub total_tokens: u64,
    pub max_tokens_per_window: u64,
    pub scratch_fixed_bytes: u64,
    pub scratch_bytes_per_token: u64,
    pub scratch_budget_bytes: u64,
    /// Already committed memory that must not be offered to this scheduler.
    pub outstanding_reservations_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct TokenWindows {
    next: u64,
    total: u64,
    width: u64,
    fixed: u64,
    per_token: u64,
}

impl TokenWindows {
    /// Reject impossible budgets, unknown/zero work and arithmetic overflow.
    /// Caller must hold the underlying budget reservation for the entire work window.
    pub fn plan(limits: TokenWindowLimits) -> Result<Self, &'static str> {
        if limits.total_tokens == 0 || limits.max_tokens_per_window == 0 {
            return Err("token count and window ceiling must be positive");
        }
        if limits.scratch_bytes_per_token == 0 {
            return Err("per-token scratch estimate must be positive");
        }
        let available = limits.scratch_budget_bytes
            .checked_sub(limits.outstanding_reservations_bytes)
            .ok_or("existing reservations exceed scratch budget")?;
        let variable = available.checked_sub(limits.scratch_fixed_bytes)
            .ok_or("fixed scratch exceeds available budget")?;
        let capacity = variable / limits.scratch_bytes_per_token;
        if capacity == 0 {
            return Err("even one token exceeds scratch budget");
        }
        let width = capacity.min(limits.max_tokens_per_window).min(limits.total_tokens);
        // All emitted windows are at most width, so validate the maximum allocation once.
        limits.scratch_bytes_per_token.checked_mul(width)
            .and_then(|n| n.checked_add(limits.scratch_fixed_bytes))
            .ok_or("window reservation overflow")?;
        Ok(Self {
            next: 0,
            total: limits.total_tokens,
            width,
            fixed: limits.scratch_fixed_bytes,
            per_token: limits.scratch_bytes_per_token,
        })
    }

    /// Cancellation is checked before yielding work, including after the last window.
    /// A cancelled caller must not schedule further native execution.
    pub fn next_checked(&mut self, cancelled: &AtomicBool) -> Result<Option<TokenWindow>, &'static str> {
        if cancelled.load(Ordering::Acquire) {
            return Err("token scheduling cancelled");
        }
        if self.next == self.total {
            return Ok(None);
        }
        let remaining = self.total - self.next;
        let count = remaining.min(self.width);
        let end = self.next.checked_add(count).ok_or("token window end overflow")?;
        let reserved_bytes = self.per_token.checked_mul(count)
            .and_then(|n| n.checked_add(self.fixed))
            .ok_or("token window allocation overflow")?;
        let window = TokenWindow { start: self.next, end, reserved_bytes };
        self.next = end;
        Ok(Some(window))
    }

    pub fn remaining_tokens(&self) -> u64 {
        self.total - self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> TokenWindowLimits {
        TokenWindowLimits {
            total_tokens: 10,
            max_tokens_per_window: 8,
            scratch_fixed_bytes: 20,
            scratch_bytes_per_token: 10,
            scratch_budget_bytes: 65,
            outstanding_reservations_bytes: 5,
        }
    }
    #[test]
    fn partitions_without_gaps_and_accounts_for_each_window() {
        let mut p = TokenWindows::plan(limits()).unwrap();
        let cancelled = AtomicBool::new(false);
        let mut windows = Vec::new();
        while let Some(w) = p.next_checked(&cancelled).unwrap() {
            assert!(w.reserved_bytes <= 60);
            windows.push((w.start, w.end, w.reserved_bytes));
        }
        assert_eq!(windows, vec![(0, 4, 60), (4, 8, 60), (8, 10, 40)]);
        assert_eq!(p.remaining_tokens(), 0);
    }
    #[test]
    fn cancellation_never_advances_cursor() {
        let mut p = TokenWindows::plan(limits()).unwrap();
        let cancelled = AtomicBool::new(false);
        assert_eq!(p.next_checked(&cancelled).unwrap().unwrap().end, 4);
        cancelled.store(true, Ordering::Release);
        assert_eq!(p.next_checked(&cancelled), Err("token scheduling cancelled"));
        assert_eq!(p.remaining_tokens(), 6);
    }
    #[test]
    fn exact_one_token_boundary() {
        let mut l = limits();
        l.scratch_budget_bytes = 35;
        l.outstanding_reservations_bytes = 5;
        let mut p = TokenWindows::plan(l).unwrap();
        assert_eq!(p.next_checked(&AtomicBool::new(false)).unwrap().unwrap(),
            TokenWindow { start: 0, end: 1, reserved_bytes: 30 });
        l.scratch_budget_bytes = 34;
        assert!(TokenWindows::plan(l).is_err());
    }
    #[test]
    fn rejects_invalid_and_overcommitted_budgets() {
        let mut l = limits();
        l.total_tokens = 0;
        assert!(TokenWindows::plan(l).is_err());
        l = limits();
        l.max_tokens_per_window = 0;
        assert!(TokenWindows::plan(l).is_err());
        l = limits();
        l.scratch_bytes_per_token = 0;
        assert!(TokenWindows::plan(l).is_err());
        l = limits();
        l.outstanding_reservations_bytes = 66;
        assert!(TokenWindows::plan(l).is_err());
    }
    #[test]
    fn huge_token_count_never_overflows_window_end() {
        let l = TokenWindowLimits {
            total_tokens: u64::MAX,
            max_tokens_per_window: u64::MAX,
            scratch_fixed_bytes: 0,
            scratch_bytes_per_token: 1,
            scratch_budget_bytes: u64::MAX,
            outstanding_reservations_bytes: 0,
        };
        let mut p = TokenWindows::plan(l).unwrap();
        assert_eq!(p.next_checked(&AtomicBool::new(false)).unwrap().unwrap().end, u64::MAX);
        assert!(p.next_checked(&AtomicBool::new(false)).unwrap().is_none());
    }
    #[test]
    fn reservations_reduce_admitted_window_width() {
        let mut l = limits();
        l.outstanding_reservations_bytes = 25;
        let mut p = TokenWindows::plan(l).unwrap();
        let w = p.next_checked(&AtomicBool::new(false)).unwrap().unwrap();
        assert_eq!((w.start, w.end, w.reserved_bytes), (0, 2, 40));
    }
}
