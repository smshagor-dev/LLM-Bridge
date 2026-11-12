//! Deterministic memory admission boundary tests; no model weights required.
use llmbridge_core::hardware::ProcessMemory;
use llmbridge_core::runtime_memory::RuntimeMemoryLimits;
use std::cell::Cell;
use std::io;

fn sample(rss: Option<u64>, cgroup: Option<u64>) -> ProcessMemory {
    ProcessMemory {
        rss_bytes: rss,
        cgroup_current_bytes: cgroup,
        ..ProcessMemory::default()
    }
}

#[test]
fn cgroup_headroom_rejects_before_native_callback() {
    let limits = RuntimeMemoryLimits {
        max_process_rss_bytes: 500,
        max_cgroup_current_bytes: Some(200),
    };
    let called = Cell::new(false);
    let result = limits.around_growth_with_sampler(
        40,
        20,
        || sample(Some(100), Some(150)),
        || {
            called.set(true);
            Ok(())
        },
    );
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
    assert!(!called.get());
}

#[test]
fn cgroup_postflight_detects_growth_even_after_native_error() {
    let limits = RuntimeMemoryLimits {
        max_process_rss_bytes: 500,
        max_cgroup_current_bytes: Some(200),
    };
    let samples = Cell::new(0);
    let result: io::Result<()> = limits.around_growth_with_sampler(
        20,
        10,
        || {
            let count = samples.get();
            samples.set(count + 1);
            sample(Some(100), Some(if count == 0 { 100 } else { 201 }))
        },
        || Err(io::Error::new(io::ErrorKind::InvalidData, "native error")),
    );
    assert_eq!(samples.get(), 2);
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
}

#[test]
fn exact_headroom_boundary_succeeds_and_returns_native_result() {
    let limits = RuntimeMemoryLimits {
        max_process_rss_bytes: 200,
        max_cgroup_current_bytes: Some(300),
    };
    let result = limits.around_growth_with_sampler(
        40,
        10,
        || sample(Some(150), Some(250)),
        || Ok(42),
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn unknown_cgroup_counter_fails_closed_before_callback() {
    let limits = RuntimeMemoryLimits {
        max_process_rss_bytes: 200,
        max_cgroup_current_bytes: Some(300),
    };
    let called = Cell::new(false);
    let result = limits.around_growth_with_sampler(
        10,
        10,
        || sample(Some(100), None),
        || {
            called.set(true);
            Ok(())
        },
    );
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
    assert!(!called.get());
}

#[test]
fn process_growth_overflow_fails_before_native_callback() {
    let limits = RuntimeMemoryLimits {
        max_process_rss_bytes: u64::MAX,
        max_cgroup_current_bytes: None,
    };
    let called = Cell::new(false);
    let result = limits.around_growth_with_sampler(
        2,
        0,
        || sample(Some(u64::MAX - 1), None),
        || {
            called.set(true);
            Ok(())
        },
    );
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
    assert!(!called.get());
}

#[test]
fn cgroup_growth_overflow_fails_before_native_callback() {
    let limits = RuntimeMemoryLimits {
        max_process_rss_bytes: u64::MAX,
        max_cgroup_current_bytes: Some(u64::MAX),
    };
    let called = Cell::new(false);
    let result = limits.around_growth_with_sampler(
        3,
        1,
        || sample(Some(10), Some(u64::MAX - 2)),
        || {
            called.set(true);
            Ok(())
        },
    );
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
    assert!(!called.get());
}
