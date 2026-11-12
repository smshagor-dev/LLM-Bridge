use llmbridge_core::inference;
use std::path::Path;

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("usage: model_contract <real-model.gguf>");
    let model = Path::new(&model);
    let mut pieces = 0usize;
    let cancelled = inference::generate(model, "The sky is", 128, 16, 2, |_| {
        pieces += 1;
        false
    });
    assert_eq!(pieces, 1, "real model must emit at least one token piece");
    assert_eq!(cancelled.unwrap_err(), "generation cancelled");

    let giant_prompt = "The sky is ".repeat(300);
    let mut generated = 0;
    let rejected = inference::generate(model, &giant_prompt, 64, 16, 2, |_| {
        generated += 1;
        true
    });
    assert_eq!(
        generated, 0,
        "oversized prompt must not generate partial output"
    );
    assert!(rejected.unwrap_err().contains("context"));
    println!("real-model cancellation and oversized prompt refusal: PASS");
}
