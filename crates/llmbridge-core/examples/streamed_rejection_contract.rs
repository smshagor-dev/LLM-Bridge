//! Deterministic negative-path contract for the streamed decoder's GGUF
//! admission. Builds minimal in-memory GGUF fixtures (no model download) and
//! asserts that `inspect_streamed_shape` (which constructs the native Decoder)
//! accepts a valid minimal llama model and rejects each malformed/unsupported
//! variant with a safe error instead of a crash. Run:
//!   cargo run --example streamed_rejection_contract --features native
use llmbridge_core::streamed_plan::inspect_streamed_shape;
use std::sync::atomic::AtomicBool;

// ---- minimal GGUF v3 writer -------------------------------------------------

const GGUF_U32: u32 = 4;
const GGUF_F32: u32 = 6;
const GGUF_STR: u32 = 8;
const GGUF_ARR: u32 = 9;
const ALIGN: u64 = 32;

enum Val {
    U32(u32),
    F32(f32),
    Str(String),
    StrArr(Vec<String>),
}

struct Tensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
}

/// (block size in elements, bytes per block) for the ggml types used by fixtures.
fn row_layout(ggml_type: u32) -> (u64, u64) {
    match ggml_type {
        0 => (1, 4),     // F32
        1 => (1, 2),     // F16
        8 => (32, 34),   // Q8_0
        9 => (32, 40),   // Q8_1
        10 => (256, 84), // Q2_K
        20 => (32, 18),  // IQ4_NL (indexable geometry; not decodable by the backend)
        _ => (1, 4),
    }
}

impl Tensor {
    fn bytes(&self) -> u64 {
        let (block, block_bytes) = row_layout(self.ggml_type);
        let ne0 = self.dims[0];
        if ne0 % block != 0 {
            // Geometry-violation fixture: reserve a block so the file has data;
            // the row-layout check rejects it before any payload is consumed.
            return self.dims.iter().product::<u64>().max(1) * 4;
        }
        let rows: u64 = self.dims[1..].iter().product::<u64>().max(1);
        (ne0 / block) * block_bytes * rows
    }
}

/// A tweakable minimal llama model; `build()` serializes it to GGUF bytes.
struct Model {
    kv: Vec<(String, Val)>,
    tensors: Vec<Tensor>,
}

fn gstr(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
}

impl Model {
    fn build(&self) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            gstr(&mut b, k);
            match v {
                Val::U32(n) => {
                    b.extend_from_slice(&GGUF_U32.to_le_bytes());
                    b.extend_from_slice(&n.to_le_bytes());
                }
                Val::F32(n) => {
                    b.extend_from_slice(&GGUF_F32.to_le_bytes());
                    b.extend_from_slice(&n.to_le_bytes());
                }
                Val::Str(s) => {
                    b.extend_from_slice(&GGUF_STR.to_le_bytes());
                    gstr(&mut b, s);
                }
                Val::StrArr(items) => {
                    b.extend_from_slice(&GGUF_ARR.to_le_bytes());
                    b.extend_from_slice(&GGUF_STR.to_le_bytes());
                    b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for it in items {
                        gstr(&mut b, it);
                    }
                }
            }
        }
        // Tensor infos with cumulative 32-aligned relative offsets.
        let mut rel: u64 = 0;
        let mut offsets = Vec::new();
        for t in &self.tensors {
            offsets.push(rel);
            rel += t.bytes().div_ceil(ALIGN) * ALIGN;
        }
        let total_data = rel;
        for (t, off) in self.tensors.iter().zip(&offsets) {
            gstr(&mut b, &t.name);
            b.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&t.ggml_type.to_le_bytes());
            b.extend_from_slice(&off.to_le_bytes());
        }
        while b.len() as u64 % ALIGN != 0 {
            b.push(0);
        }
        b.extend_from_slice(&vec![0u8; total_data as usize]);
        b
    }
}

/// Valid minimal 1-layer, tied-embedding llama model: dim 32, hidden 64,
/// 1 head, head_dim 32, vocab 4. All weights F32.
fn valid_llama() -> Model {
    let tensors = vec![
        t("token_embd.weight", &[32, 4]),
        t("output_norm.weight", &[32]),
        t("blk.0.attn_norm.weight", &[32]),
        t("blk.0.attn_q.weight", &[32, 32]),
        t("blk.0.attn_k.weight", &[32, 32]),
        t("blk.0.attn_v.weight", &[32, 32]),
        t("blk.0.attn_output.weight", &[32, 32]),
        t("blk.0.ffn_norm.weight", &[32]),
        t("blk.0.ffn_gate.weight", &[32, 64]),
        t("blk.0.ffn_up.weight", &[32, 64]),
        t("blk.0.ffn_down.weight", &[64, 32]),
    ];
    Model {
        kv: vec![
            ("general.architecture".into(), Val::Str("llama".into())),
            ("general.alignment".into(), Val::U32(32)),
            ("llama.embedding_length".into(), Val::U32(32)),
            ("llama.feed_forward_length".into(), Val::U32(64)),
            ("llama.block_count".into(), Val::U32(1)),
            ("llama.attention.head_count".into(), Val::U32(1)),
            ("llama.attention.head_count_kv".into(), Val::U32(1)),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                Val::F32(1e-5),
            ),
            ("llama.rope.dimension_count".into(), Val::U32(32)),
            ("llama.rope.freq_base".into(), Val::F32(10000.0)),
            ("llama.context_length".into(), Val::U32(64)),
            (
                "tokenizer.ggml.tokens".into(),
                Val::StrArr(vec!["a".into(), "b".into(), "c".into(), "d".into()]),
            ),
        ],
        tensors,
    }
}

fn t(name: &str, dims: &[u64]) -> Tensor {
    Tensor {
        name: name.into(),
        dims: dims.to_vec(),
        ggml_type: 0,
    }
}

fn set_kv(m: &mut Model, key: &str, v: Val) {
    if let Some(slot) = m.kv.iter_mut().find(|(k, _)| k == key) {
        slot.1 = v;
    } else {
        m.kv.push((key.into(), v));
    }
}

fn write_temp(bytes: &[u8], tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "llmbridge-reject-{}-{}.gguf",
        std::process::id(),
        tag
    ));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn expect_ok(m: &Model, tag: &str) {
    let p = write_temp(&m.build(), tag);
    let cancelled = AtomicBool::new(false);
    let r = inspect_streamed_shape(&p, 64 * 1024, &cancelled);
    assert!(r.is_ok(), "[{tag}] expected accept, got {:?}", r.err());
    std::fs::remove_file(&p).ok();
    println!("  ACCEPT  {tag}");
}

fn expect_reject(m: &Model, tag: &str, needle: &str) {
    let p = write_temp(&m.build(), tag);
    let cancelled = AtomicBool::new(false);
    let r = inspect_streamed_shape(&p, 64 * 1024, &cancelled);
    let err = r.expect_err(&format!("[{tag}] expected rejection but it was accepted"));
    assert!(
        err.contains(needle),
        "[{tag}] error {err:?} did not contain {needle:?}"
    );
    std::fs::remove_file(&p).ok();
    println!("  REJECT  {tag}: {err}");
}

fn main() {
    println!("streamed_rejection_contract: positive baseline (tied embeddings)");
    expect_ok(&valid_llama(), "valid-llama-tied");

    println!("positive: untied output projection (separate output.weight)");
    let mut m = valid_llama();
    m.tensors.push(t("output.weight", &[32, 4]));
    expect_ok(&m, "valid-llama-untied");

    println!("negative: untied output.weight with wrong vocabulary dimension");
    let mut m = valid_llama();
    m.tensors.push(t("output.weight", &[32, 5]));
    expect_reject(
        &m,
        "bad-untied-output",
        "unsupported streamed tensor type/shape",
    );

    println!("negative: unsupported architecture");
    let mut m = valid_llama();
    set_kv(&mut m, "general.architecture", Val::Str("mistral".into()));
    expect_reject(&m, "unknown-arch", "architecture");

    println!("negative: unknown architecture metadata key");
    let mut m = valid_llama();
    set_kv(&mut m, "llama.attention.sliding_window", Val::U32(4096));
    expect_reject(
        &m,
        "unknown-key",
        "unsupported architecture metadata option",
    );

    println!("negative: unsupported runtime RoPE scaling metadata");
    let mut m = valid_llama();
    set_kv(&mut m, "llama.rope.scaling.type", Val::Str("yarn".into()));
    expect_reject(
        &m,
        "rope-scaling",
        "unsupported architecture metadata option",
    );

    println!("negative: key_length != head_dim");
    let mut m = valid_llama();
    set_kv(&mut m, "llama.attention.key_length", Val::U32(16));
    expect_reject(&m, "bad-key-length", "key/value length == head dim");

    println!("negative: vocab_size disagrees with tokenizer");
    let mut m = valid_llama();
    set_kv(&mut m, "llama.vocab_size", Val::U32(7));
    expect_reject(&m, "bad-vocab-size", "vocab_size disagrees");

    println!("negative: invalid GQA ratio (head_count indivisible)");
    let mut m = valid_llama();
    set_kv(&mut m, "llama.attention.head_count", Val::U32(3));
    expect_reject(&m, "bad-gqa", "configuration exceeds supported limits");

    println!("negative: partial-head rotary dimension");
    let mut m = valid_llama();
    set_kv(&mut m, "llama.rope.dimension_count", Val::U32(16));
    expect_reject(&m, "partial-rope", "full-head RoPE");

    println!("negative: extra unexpected tensor");
    let mut m = valid_llama();
    m.tensors.push(t("blk.0.adapter.weight", &[32, 32]));
    expect_reject(&m, "extra-tensor", "unsupported extra tensors");

    println!("negative: wrong output/embedding row count (vocab mismatch)");
    let mut m = valid_llama();
    m.tensors[0].dims = vec![32, 5]; // token_embd rows != 4 tokenizer tokens
    expect_reject(
        &m,
        "bad-embd-shape",
        "unsupported streamed tensor type/shape",
    );

    println!("negative: unsupported tensor element type (F16, valid geometry)");
    let mut m = valid_llama();
    m.tensors[3].ggml_type = 1; // F16: passes the index block check, not in the decoder allowlist
    expect_reject(
        &m,
        "unsupported-type",
        "unsupported streamed tensor type/shape",
    );

    println!("negative: quantization geometry violation (Q2_K on 32-wide row)");
    let mut m = valid_llama();
    m.tensors[3].ggml_type = 10; // Q2_K needs 256-divisible rows; 32 is not
    expect_reject(&m, "bad-quant-geometry", "block layout");

    println!("negative: missing mandatory metadata (embedding_length)");
    let mut m = valid_llama();
    m.kv.retain(|(k, _)| k != "llama.embedding_length");
    expect_reject(
        &m,
        "missing-embd",
        "missing/invalid streamed model integer metadata",
    );

    // ---- GPT-OSS 120B (MoE / MXFP4) explicit unsupported-architecture gate ----
    // The streamed decoder is a dense llama/qwen2 path. GPT-OSS is a Mixture-of-
    // Experts model with MXFP4 (ggml type 39) weights. These cases pin the three
    // independent reasons GPT-OSS is refused BEFORE any 63 GB download, so the
    // "unsupported" contract is enforced rather than asserted in prose.

    println!("negative: GPT-OSS architecture identifier is refused");
    let mut m = valid_llama();
    set_kv(&mut m, "general.architecture", Val::Str("gpt-oss".into()));
    expect_reject(&m, "gpt-oss-arch", "llama and qwen2 architectures only");

    println!("negative: MoE expert_count metadata is refused (no router path)");
    let mut m = valid_llama();
    set_kv(&mut m, "llama.expert_count", Val::U32(128));
    expect_reject(
        &m,
        "moe-expert-count",
        "unsupported architecture metadata option",
    );

    println!("negative: MXFP4 tensor type (ggml type 39) is refused at index");
    let mut m = valid_llama();
    m.tensors[3].ggml_type = 39; // MXFP4: absent from tensor_layout's admitted set
    expect_reject(&m, "mxfp4-type", "not supported for indexed range reads");

    // ---- NVIDIA Nemotron 3 Super 120B-A12B (nemotron_h_moe) unsupported gate ----
    // The real downloaded model (UD-IQ2_XXS, 3 shards) declares architecture
    // "nemotron_h_moe": a hybrid Mamba-2 state-space + attention + 512-expert latent
    // MoE design. The streamed backend is a dense llama/qwen2 decoder with none of
    // that math. These cases pin that it is refused for INFERENCE even though the
    // index can now describe its i-quant tensor geometry (see below) -- indexing a
    // tensor's bytes is not the same as being able to decode and run it.

    println!("negative: nemotron_h_moe architecture identifier is refused");
    let mut m = valid_llama();
    set_kv(
        &mut m,
        "general.architecture",
        Val::Str("nemotron_h_moe".into()),
    );
    expect_reject(&m, "nemotron-arch", "llama and qwen2 architectures only");

    println!("negative: hybrid SSM metadata option under the active arch is refused");
    let mut m = valid_llama();
    // A Mamba-2 state-space key the dense decoder does not implement; it must be
    // rejected, never silently ignored/approximated.
    set_kv(&mut m, "llama.ssm.state_size", Val::U32(128));
    expect_reject(
        &m,
        "nemotron-ssm-key",
        "unsupported architecture metadata option",
    );

    println!("negative: IQ4_NL tensor is indexable but NOT decodable by the backend");
    let mut m = valid_llama();
    // IQ4_NL (ggml type 20) now has valid 32-element block geometry in the index,
    // so it passes range-read admission -- but the streamed decoder's supported
    // compute types do not include it, so execution is still refused. This is the
    // honest boundary between "can index/stream bytes" and "can run the kernel".
    m.tensors[3].ggml_type = 20;
    expect_reject(
        &m,
        "iq4nl-not-decodable",
        "unsupported streamed tensor type/shape",
    );

    println!("ALL REJECTION CONTRACTS PASSED");
}
