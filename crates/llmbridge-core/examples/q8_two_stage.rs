//! Example: compose two real GGUF Q8_0 matrices through bounded native reads.
//!
//! cargo run --release -p llmbridge-core --features native --example q8_two_stage -- \\
//!   /models/two-layer.gguf first.weight second.weight 1,1,1,1
//!
//! The comma-separated input must match the first tensor's actual column
//! count (32-16384, multiple of 32). This is not a complete model runtime.
use llmbridge_core::gguf_index::GgufTensorIndex;
use llmbridge_core::q8_chain::q8_0_two_stage_with_budget;
use llmbridge_core::storage::BoundedLocalReader;
use std::sync::atomic::AtomicBool;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: q8_two_stage <model.gguf> <first-tensor> <second-tensor> <comma-separated-floats>");
        std::process::exit(2);
    }
    let input: Vec<f32> = args[4]
        .split(',')
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()?;
    let index = GgufTensorIndex::inspect(&args[1])?;
    let mut reader = BoundedLocalReader::open(&args[1], 128 * 1024, 256 * 1024)?;
    let operator_budget = 512 * 1024;
    // The wrapper additionally reserves space for the first stage output,
    // which is bounded by native Q8_0 rows <= 65536 (256 KiB of f32).
    let graph_budget = operator_budget + 256 * 1024;
    let result = q8_0_two_stage_with_budget(
        &index,
        &mut reader,
        &args[2],
        &args[3],
        &input,
        &AtomicBool::new(false),
        operator_budget,
        graph_budget,
    ).map_err(std::io::Error::other)?;
    eprintln!("tensor bytes read: {}", reader.disk_bytes_read());
    for value in result {
        println!("{value:.9}");
    }
    Ok(())
}
