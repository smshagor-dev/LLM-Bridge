# Cooperative FIFO memory admission

The `AdmissionQueue` provides bounded pending-request slots and FIFO ordering
on top of the existing atomic `MemoryBudget`. Enqueueing does not allocate a
reservation. The head ticket receives a reservation only if the shared budget
can cover it. Later tickets cannot overtake a blocked head. Callers must
explicitly cancel abandoned tickets and hold returned `Reservation` values
until native execution completes.

This is an **opt-in library primitive**, not wired into the streamed decoder
entry point or CLI. It does not spawn threads, guarantee liveness, measure
physical memory, or enforce native RSS. The queue does not shrink the OS/cgroup
admission requirements and cannot replace model authentication, cancellation
checks, or the existing process-memory guards.

Regression tests cover FIFO blocking, cancellation, bounded queue capacity,
invalid requests and competing admission polls. No physical hardware
performance or 2 GB viability is asserted.
