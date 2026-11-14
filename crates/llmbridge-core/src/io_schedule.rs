//! Deterministic, allocation-free I/O window admission for streamed CPU stages.
//! This plans caller-owned staging only. It does not allocate, read or enforce RSS.
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoWindowRequest {
    /// Bytes left in the authenticated tensor range.
    pub remaining_bytes: u64,
    /// Maximum permitted bytes per bounded source read.
    pub source_chunk_bytes: usize,
    /// Tensor block alignment; a final partial block is never admitted.
    pub block_bytes: usize,
    /// Bytes available for new staging after all other reservations.
    pub staging_budget_bytes: usize,
    /// Upper limit on concurrent buffers.
    pub max_in_flight: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoWindowPlan {
    pub chunk_bytes: usize,
    pub in_flight: usize,
    pub staging_charge_bytes: usize,
    pub covered_bytes: u64,
}

fn invalid(reason: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, reason)
}

/// Return a bounded, block-aligned window without rounding the source range up.
/// A caller must reserve `staging_charge_bytes` before allocating any buffers,
/// authenticate the source session, and recheck cancellation before every read.
pub fn plan_io_window(request: IoWindowRequest) -> io::Result<IoWindowPlan> {
    if request.remaining_bytes == 0
        || request.source_chunk_bytes == 0
        || request.block_bytes == 0
        || request.max_in_flight == 0
    {
        return Err(invalid("empty range or invalid I/O window bounds"));
    }
    if request.remaining_bytes % request.block_bytes as u64 != 0 {
        return Err(invalid("tensor range is not block aligned"));
    }
    let maximum = request.source_chunk_bytes
        .min(request.staging_budget_bytes)
        .min(usize::try_from(request.remaining_bytes).unwrap_or(usize::MAX));
    let chunk = maximum - maximum % request.block_bytes;
    if chunk == 0 {
        return Err(invalid("insufficient staging budget for one tensor block"));
    }
    let possible = request.staging_budget_bytes / chunk;
    let by_range = request.remaining_bytes / chunk as u64;
    let slots = possible.min(request.max_in_flight)
        .min(usize::try_from(by_range).unwrap_or(usize::MAX));
    if slots == 0 {
        return Err(invalid("no complete I/O window fits"));
    }
    let charge = chunk.checked_mul(slots)
        .ok_or_else(|| invalid("staging charge overflow"))?;
    let covered = (chunk as u64).checked_mul(slots as u64)
        .ok_or_else(|| invalid("window range overflow"))?;
    if charge > request.staging_budget_bytes || covered > request.remaining_bytes {
        return Err(invalid("I/O window exceeds admitted bounds"));
    }
    Ok(IoWindowPlan {
        chunk_bytes: chunk,
        in_flight: slots,
        staging_charge_bytes: charge,
        covered_bytes: covered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> IoWindowRequest {
        IoWindowRequest {
            remaining_bytes: 1024,
            source_chunk_bytes: 256,
            block_bytes: 32,
            staging_budget_bytes: 640,
            max_in_flight: 4,
        }
    }

    #[test]
    fn charges_every_concurrent_staging_buffer() {
        let p = plan_io_window(request()).unwrap();
        assert_eq!((p.chunk_bytes, p.in_flight, p.staging_charge_bytes, p.covered_bytes),
                   (256, 2, 512, 512));
    }

    #[test]
    fn pressure_reduces_window_without_overcommitting() {
        let p = plan_io_window(IoWindowRequest {
            staging_budget_bytes: 80, ..request()
        }).unwrap();
        assert_eq!((p.chunk_bytes, p.in_flight, p.staging_charge_bytes), (64, 1, 64));
    }

    #[test]
    fn tiny_tail_is_not_rounded_beyond_tensor_end() {
        let p = plan_io_window(IoWindowRequest {
            remaining_bytes: 32, ..request()
        }).unwrap();
        assert_eq!((p.chunk_bytes, p.in_flight, p.covered_bytes), (32, 1, 32));
    }

    #[test]
    fn rejects_misaligned_range_and_unserviceable_budget() {
        for r in [
            IoWindowRequest { remaining_bytes: 33, ..request() },
            IoWindowRequest { staging_budget_bytes: 31, ..request() },
            IoWindowRequest { block_bytes: 0, ..request() },
            IoWindowRequest { max_in_flight: 0, ..request() },
            IoWindowRequest { remaining_bytes: 0, ..request() },
        ] {
            assert!(plan_io_window(r).is_err());
        }
    }

    #[test]
    fn huge_range_and_budget_remain_checked() {
        let p = plan_io_window(IoWindowRequest {
            remaining_bytes: u64::MAX - 63,
            source_chunk_bytes: 1024,
            block_bytes: 64,
            staging_budget_bytes: usize::MAX,
            max_in_flight: usize::MAX,
        }).unwrap();
        assert_eq!(p.chunk_bytes, 1024);
        assert!(p.covered_bytes <= u64::MAX - 63);
        assert!(p.staging_charge_bytes <= usize::MAX);
    }

    #[test]
    fn source_limit_and_slot_limit_are_both_respected() {
        let p = plan_io_window(IoWindowRequest {
            source_chunk_bytes: 70,
            max_in_flight: 1,
            ..request()
        }).unwrap();
        assert_eq!((p.chunk_bytes, p.in_flight, p.staging_charge_bytes), (64, 1, 64));
    }
}
