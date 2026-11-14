# Pausing queued inference admission

`AdmissionQueue::set_paused(bool)` changes whether pending requests can acquire a reservation. It returns the previous state; repeated pause/resume calls are idempotent. `is_paused()` is a synchronized snapshot. The default is unpaused, preserving existing behavior.

Pause and reservation acquisition use the same queue mutex. Once a pause operation completes, subsequent admission polls cannot reserve memory until another controller resumes the queue. Requests admitted before the pause retain their reservations and may start or continue native work afterwards. This is admission control, not a barrier that freezes all execution, cancels sessions or reclaims active memory.

While paused, the existing pending-slot limit, validation, FIFO order, cancellation and pending-handle RAII stay active. New requests can queue up to the configured limit and get QueueFull beyond it. Unknown, cancelled and foreign-queue tickets remain errors. Resume does not skip the oldest live request or bypass memory-capacity checks. Already-admitted reservations are released only through their existing ownership lifetime.

`generate_streamed_queued` already polls `try_admit`, so the control gates actual native generation without adding a thread or a new FFI entry point. Waiting still checks the caller's cancellation flag and parks between polls. Resume latency depends on that polling/OS scheduling; there is no wake-up deadline. An external application can use this API when it decides to stop accepting more memory-intensive work, but this change adds no automatic pressure sampling, CLI pause command or global process/RSS enforcement. Separate queues keep separate pause states. Controllers must coordinate pause ownership; the latest explicit state setter wins.

This independent change targets main's public queued-generation API. The worker executor in #84 owns a private queue; exposing equivalent controller methods there is separate integration work.

## Real validation

The pinned Q8 contract first obtains direct complete-output/all-FP32-logit hashes. Two requests enqueue while paused with all memory capacity free and zero reserved bytes. After resume the FIFO head starts, pauses admission from its first-logits consumer, and completes its full native response. The second request stays pending after that response has released all memory. A second resume lets it complete with the same output/logit hashes. Cancellation of a paused missing-model request must remove its ticket without opening the model, taking a reservation or implicitly resuming the queue.

Unit regressions cover paused queue bounds, FIFO resume, state idempotence, existing reservation preservation, pending-handle cleanup and invalid/cancelled/foreign tickets. Reproduce with `cargo run --release -p llmbridge-core --features native --example paused_admission_contract -- models/stories15M-q8_0.gguf`.

These are cooperative admission/correctness tests. They do not measure physical memory reclamation, throughput, real 2GB hardware or large-model feasibility. Native numerical code, model identity/authentication and process limits are unchanged.
