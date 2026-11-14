# Cooperative FIFO memory admission

The `AdmissionQueue` provides bounded pending-request slots and FIFO ordering
on top of the existing atomic `MemoryBudget`. Enqueueing does not allocate a
reservation. The head ticket receives a reservation only if the shared budget
can cover it. Later tickets cannot overtake a blocked head. Callers must
explicitly cancel abandoned tickets and hold returned `Reservation` values
until native execution completes.

The opt-in `generate_streamed_queued` wrapper now connects this queue to real
synchronous generation; see [queued streamed inference](QUEUED_STREAMED_INFERENCE.md).
It is not wired into the CLI. It does not spawn threads, guarantee liveness, measure
physical memory, or enforce native RSS. The queue does not shrink the OS/cgroup
admission requirements and cannot replace model authentication, cancellation
checks, or the existing process-memory guards.

Regression tests cover FIFO blocking, cancellation, bounded queue capacity,
invalid requests and competing admission polls. No physical hardware
performance or 2 GB viability is asserted.

Tickets include queue identity; passing a ticket to another queue cannot cancel or admit its request. Use `request()` for an owned pending handle that cleans up on drop. Low-level `enqueue()` retains explicit cancellation semantics.
