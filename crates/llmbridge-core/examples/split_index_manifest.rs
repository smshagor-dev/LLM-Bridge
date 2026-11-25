//! Logical multi-shard (split) GGUF manifest.
//!
//! Validates and indexes a set of split GGUF shards WITHOUT reading any tensor
//! payload: it parses each shard's header, metadata and tensor table, checks the
//! `split.*` consistency (count, per-shard `split.no`, aggregate tensor count),
//! maps every tensor to its owning shard and byte range, and prints a manifest.
//!
//! This is metadata-only inspection, not inference and not integrity proof of the
//! weights (hash the shards separately for that). Usage:
//!   cargo run --example split_index_manifest -- shard-00001.gguf shard-00002.gguf ...
use llmbridge_core::gguf_index::SplitGgufIndex;
use std::collections::BTreeMap;

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        8 => "Q8_0",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        16 => "IQ2_XXS",
        20 => "IQ4_NL",
        30 => "BF16",
        _ => "other",
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: split_index_manifest <shard.gguf> [shard.gguf ...]");
        std::process::exit(2);
    }
    let index = match SplitGgufIndex::inspect(&paths) {
        Ok(index) => index,
        Err(error) => {
            eprintln!("split index REJECTED: {error}");
            std::process::exit(1);
        }
    };
    println!("split index ACCEPTED");
    println!("  split_count      = {}", index.split_count);
    println!("  split.tensors    = {}", index.total_tensors);
    println!("  indexed tensors  = {}", index.indexed_tensor_count());
    let mut type_hist: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut indexed_bytes: u64 = 0;
    for member in &index.members {
        let (tcount, mbytes) = match &member.index {
            None => (0u64, 0u64),
            Some(idx) => {
                let mut bytes = 0u64;
                for t in &idx.tensors {
                    *type_hist.entry(ggml_type_name(t.ggml_type)).or_default() += 1;
                    bytes += t.byte_len;
                }
                (idx.tensors.len() as u64, bytes)
            }
        };
        indexed_bytes += mbytes;
        println!(
            "  shard split_no={} tensors={:>4} indexed_tensor_bytes={:>14} path={}",
            member.split_no,
            tcount,
            mbytes,
            member.path.display()
        );
    }
    println!("  type histogram (indexed tensors):");
    for (name, count) in &type_hist {
        println!("    {name:<8} {count}");
    }
    println!("  total indexed tensor bytes = {indexed_bytes}");
    // A few representative logical->shard resolutions.
    for name in [
        "token_embd.weight",
        "output.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.83.ffn_up_exps.weight",
        "blk.87.ffn_up_exps.weight",
    ] {
        if let Some((member, region)) = index.tensor(name) {
            println!(
                "  resolve {name:<28} -> shard split_no={} type={} dims={:?} file_offset={} bytes={}",
                member.split_no,
                ggml_type_name(region.ggml_type),
                region.dimensions,
                region.file_offset,
                region.byte_len
            );
        }
    }
}
