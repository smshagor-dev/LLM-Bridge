# FIFO admission for streamed inference

`generate_streamed_queued` connects the bounded FIFO queue to the existing synchronous streamed decoder. It waits without opening or authenticating the model, acquires one atomic cooperative reservation, and holds it through model inspection/authentication, native execution and every consumer callback. Success, source/native failure, consumer cancellation and Rust unwinding release it through RAII. Dropping a still-pending request cancels its ticket. Existing generation entry points keep their behavior.

```rust
use llmbridge_core::{
    admission_queue::AdmissionQueue, memory_budget::MemoryBudget,
    queued_streamed::{generate_streamed_queued, QueuedStreamedRequest},
};
use std::{path::Path, sync::atomic::AtomicBool};
let request = QueuedStreamedRequest::new(Path::new("model.gguf"), "Example");
let budget = MemoryBudget::new(request.reservation_bytes()?)?;
let queue = AdmissionQueue::new(budget, 32)?;
let result = generate_streamed_queued(&queue, request, &AtomicBool::new(false),
    |_| true, |_| true)?;
```

This illustrates the API; adapt application error conversions. The request's `control` carries the same optional trusted digest and cache-release signal as controlled generation. No new native ABI, weight operator or sampling code is introduced. CPU incompatibility is checked before enqueueing, model I/O or native parsing.

The charge is the configured native workspace ceiling, one Rust source chunk, exact optional-cache allocation, the configured authentication ceiling when a trusted digest is present, and the caller's additional metadata/runtime allowance (default32MiB). It is checked for arithmetic overflow and against queue capacity before waiting. The allowance is explicitly an estimate; model metadata/runtime expansion is not derived or enforced by this queue. Use a shared queue and conservatively sized capacity/reserves for competing sessions. Independent queues share capacity only when constructed with the same MemoryBudget.

Reservations count configured budgets, not actual allocations, total RSS, available system RAM or an OS memory limit. Other processes/direct nonqueued callers are outside this accounting. Admission can conservatively reject a session that would fit. The native decoder still checks its workspace, configuration, context and source bounds; explicit Linux process limits remain separate. There is no CLI/API-server worker scheduler or automatic model selection here.

Pending requests poll nonblockingly and park the caller thread for up to5ms between polls. Cancellation is checked before/while waiting and after acquisition, with a fresh native check at generation startup. This interval is not a scheduling deadline or asynchronous wake-up guarantee. Synchronously nesting a queued call inside a consumer callback can deadlock if it needs the outer call's held budget; coordinate independent workers and cancellation instead.

Tickets include an immutable process-local queue identity as well as a per-queue sequence. Cross-queue ticket values cannot cancel or acquire another queue's request, even when their sequence numbers coincide. `PendingAdmission` transfers ownership to the returned Reservation only after successful admission, so dropping the pending handle cannot release an active session's budget. Low-level manual enqueue/cancel remains available.

## Validation

Queue regressions cover cross-queue isolation, pending-handle cleanup, no double admission, fairness, cancellation and capacity bounds. Charge tests exercise exact cache geometry, authentication overhead, invalid bounds and overflow. The real Q8 fixture contract compares complete output bytes and every FP32 logit hash with direct execution, asserts that callbacks observe the full reservation, and requires zero remaining charge after successful inference, missing-model rejection and consumer cancellation. A separate thread blocked behind a full budget uses a nonexistent model path: cancellation must remove its ticket before any model-open error can occur. Oversized charges fail without queueing or allocation.

The real model remains the pinned tiny checkpoint. These are cooperative admission/correctness results, not physical2GB, whole-process RSS, throughput or large-model feasibility results. Native CPU rejection now covers the queued entry point in addition to the existing twelve paths.
