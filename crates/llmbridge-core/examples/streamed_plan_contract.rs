//! The planner and decoder agree at the exact native workspace boundary.
use llmbridge_core::{
    hardware::Hardware,
    streamed::{generate_streamed, StreamedConfig},
    streamed_plan::{inspect_streamed_shape, plan_streamed_shape, StreamedPlanRequest},
};
use std::{path::Path, sync::atomic::AtomicBool};
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: streamed_plan_contract <real.gguf>");
    let model = Path::new(&path);
    let cancelled = AtomicBool::new(false);
    let shape = inspect_streamed_shape(model, 64 * 1024, &cancelled).unwrap();
    assert_eq!(
        shape.training_context, 128,
        "protocol uses pinned stories15M"
    );
    let hardware = Hardware {
        logical_cpus: 2,
        total_ram_bytes: Some(2 * 1024 * 1024 * 1024),
        available_ram_bytes: Some(1024 * 1024 * 1024),
    };
    let budget = shape.workspace_for_context(96).unwrap();
    let request = StreamedPlanRequest {
        workspace_budget_bytes: budget,
        allow_context_reduction: true,
        ..Default::default()
    };
    let plan = plan_streamed_shape(shape, &hardware, request).unwrap();
    assert_eq!(plan.selected_context, 96);
    assert_eq!(plan.native_workspace_bytes, budget);
    assert!(shape.workspace_for_context(97).unwrap() > budget);
    let config = StreamedConfig {
        context_tokens: 96,
        max_tokens: 8,
        workspace_bytes: budget,
        ..Default::default()
    };
    let mut output = Vec::new();
    generate_streamed(
        model,
        "The sky is",
        config,
        &cancelled,
        |b| {
            output.extend_from_slice(b);
            true
        },
        |_| true,
    )
    .unwrap();
    assert!(!output.is_empty());
    let rejected = generate_streamed(
        model,
        "The sky is",
        StreamedConfig {
            workspace_bytes: budget - 1,
            ..config
        },
        &cancelled,
        |_| panic!("token beyond workspace budget"),
        |_| panic!("logits beyond workspace budget"),
    );
    assert!(rejected.unwrap_err().contains("workspace exceeds budget"));
    println!("{}", plan.to_json(&shape, &request));
    println!("exact native workspace boundary generation/refusal: PASS");
}
