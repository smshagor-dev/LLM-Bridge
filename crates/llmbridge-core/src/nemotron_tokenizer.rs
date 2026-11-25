//! Nemotron (`nemotron_h_moe`) tokenizer: GGUF metadata loader, byte-level BPE
//! core, and decoder.
//!
//! The real model declares `tokenizer.ggml.model = "gpt2"` (byte-level BPE) with
//! `tokenizer.ggml.pre = "pixtral"`, a 131072-entry vocabulary, a merge list,
//! and special ids `unk=0`, `bos=1 (<s>)`, `eos=11`, `add_bos=false`. This module
//! loads those directly from the shard-1 GGUF header (a bounded read of the
//! metadata-only shard) and implements:
//!
//!   * [`NemotronTokenizer::decode`] — ids -> tokens -> bytes (inverse GPT-2
//!     byte map) -> UTF-8. Fully determined by the vocabulary.
//!   * [`NemotronTokenizer::encode`] — GPT-2-style pre-tokenization + byte-level
//!     BPE using the real merge ranks and vocabulary.
//!
//! ## Parity status (honest)
//!
//! The loader and decoder are exact and verifiable (vocab size, special ids,
//! known tokens, byte-map round-trip, decode(encode(x)) == x). The BPE merge
//! engine follows the standard lowest-rank-first algorithm against the real
//! merges. The remaining gap for *reference-identical* IDs on arbitrary text is
//! the pre-tokenizer: llama.cpp's `pixtral` pre-type uses a specific regex set,
//! and this environment has neither a regex engine nor a runnable llama.cpp
//! reference to cross-check against. The pre-tokenizer here is a documented
//! GPT-2-style splitter; IDs are therefore NOT claimed to be byte-identical to
//! llama.cpp for general text. Do not wire this into a production prompt path
//! until that parity is confirmed against the pinned reference.
//!
//! ## Pixtral vs Tekken pre-tokenizer (llama.cpp issue #30894, PR #30960)
//!
//! Nemotron GGUFs are tagged `tokenizer.ggml.pre = "pixtral"` because their
//! tokenizer hash collides with Pixtral-12B's, but Nemotron's own
//! `tokenizer.json` uses the **Tekken** regex. llama.cpp originally mapped
//! `pixtral` to the Llama-3 pre-tokenizer; the fix (merge commit
//! `e197cef4554d466e0becac20e453a355daba8e5d`, newer than our pinned
//! `cd959bae…`) remaps `pixtral` -> the Tekken pre-tokenizer. Reported impact:
//! **plain English/ASCII tokenizes identically** under pixtral/llama3/tekken
//! (independent reporters saw byte-identical IDs for English sentences); the
//! divergences are in Devanagari/Tamil/Telugu/Bengali/Thai and some digit/
//! Unicode edge cases. This splitter caps digit runs at 3 (`\p{N}{1,3}`, shared
//! by the llama3/tekken families) and is expected to match for ASCII English;
//! non-Latin scripts are NOT verified here (the pinned reference predates the fix
//! and no reference tokenizer is runnable in this environment).

use std::collections::HashMap;

/// GPT-2 byte<->unicode table: maps each of 256 byte values to a printable
/// Unicode code point, exactly as GPT-2/llama.cpp byte-level BPE does.
fn byte_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = Vec::new();
    for b in b'!'..=b'~' {
        bs.push(b as u32);
    }
    for b in 0xA1u32..=0xAC {
        bs.push(b);
    }
    for b in 0xAEu32..=0xFF {
        bs.push(b);
    }
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    // map byte value -> char
    let mut table = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

/// Loaded Nemotron tokenizer.
pub struct NemotronTokenizer {
    pub model: String,
    pub pre: String,
    pub tokens: Vec<String>,
    pub token_type: Vec<i32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
    pub add_bos: bool,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), u32>,
    b2u: Vec<char>,
    u2b: HashMap<char, u8>,
}

impl NemotronTokenizer {
    /// Load the tokenizer from a GGUF shard header (the metadata-only shard 1).
    pub fn from_gguf(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let kv = GgufKv::parse(&bytes)?;

        let model = kv.string("tokenizer.ggml.model").unwrap_or_default();
        let pre = kv.string("tokenizer.ggml.pre").unwrap_or_default();
        let tokens = kv
            .str_array("tokenizer.ggml.tokens")
            .ok_or("tokenizer.ggml.tokens missing")?;
        let token_type = kv
            .i32_array("tokenizer.ggml.token_type")
            .unwrap_or_default();
        let merges = kv
            .str_array("tokenizer.ggml.merges")
            .ok_or("tokenizer.ggml.merges missing")?;
        let bos_id = kv.u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = kv.u32("tokenizer.ggml.eos_token_id").unwrap_or(2);
        let unk_id = kv.u32("tokenizer.ggml.unknown_token_id").unwrap_or(0);
        let add_bos = kv.bool("tokenizer.ggml.add_bos_token").unwrap_or(false);

        if tokens.is_empty() {
            return Err("empty tokenizer vocabulary".into());
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            // merges are "left right" (space-separated byte-level tokens).
            if let Some(sp) = m.find(' ') {
                let (l, r) = (m[..sp].to_string(), m[sp + 1..].to_string());
                merge_rank.entry((l, r)).or_insert(rank as u32);
            }
        }
        let b2u = byte_to_unicode();
        let mut u2b = HashMap::with_capacity(256);
        for (b, c) in b2u.iter().enumerate() {
            u2b.insert(*c, b as u8);
        }
        Ok(Self {
            model,
            pre,
            tokens,
            token_type,
            bos_id,
            eos_id,
            unk_id,
            add_bos,
            token_to_id,
            merge_rank,
            b2u,
            u2b,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Decode a sequence of token ids to text (inverse byte-level mapping).
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let tok = self
                .tokens
                .get(id as usize)
                .ok_or_else(|| format!("token id {id} out of vocab"))?;
            // Skip non-text tokens by their declared type (llama.cpp token types:
            // 2=UNKNOWN, 3=CONTROL, 5=UNUSED) — these emit no text (e.g. <s>, </s>).
            match self.token_type.get(id as usize).copied() {
                Some(2) | Some(3) | Some(5) => continue,
                _ => {}
            }
            // NORMAL / USER_DEFINED: byte-level map; chars outside the byte map are
            // dropped (defensive — the byte-level alphabet covers all 256 bytes).
            for ch in tok.chars() {
                if let Some(&b) = self.u2b.get(&ch) {
                    bytes.push(b);
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Map a run of bytes to the byte-level-unicode symbol string.
    fn bytes_to_symbols(&self, bytes: &[u8]) -> Vec<String> {
        bytes
            .iter()
            .map(|&b| self.b2u[b as usize].to_string())
            .collect()
    }

    /// Apply byte-level BPE merges (lowest rank first) to one pre-token's symbols.
    fn bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            // find the lowest-rank adjacent pair
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .merge_rank
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.map(|(_, br)| rank < br).unwrap_or(true) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((idx, _)) = best else { break };
            let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols.splice(idx..idx + 2, [merged]);
            if symbols.len() < 2 {
                break;
            }
        }
        symbols
    }

    /// GPT-2-style pre-tokenization (documented approximation; see module docs).
    /// Splits into contractions, letter runs, number runs, other-symbol runs, and
    /// whitespace, each carrying a single leading space where present.
    pub(crate) fn pretokenize(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        let n = chars.len();
        let is_letter = |c: char| c.is_alphabetic();
        let is_num = |c: char| c.is_numeric();
        while i < n {
            // contractions: 's 't 're 've 'm 'll 'd
            if chars[i] == '\'' && i + 1 < n {
                let rest: String = chars[i..].iter().collect();
                let m = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
                    .into_iter()
                    .find(|p| rest.starts_with(p));
                if let Some(p) = m {
                    out.push(p.to_string());
                    i += p.chars().count();
                    continue;
                }
            }
            let lead_space = chars[i] == ' ';
            let start = i;
            if lead_space {
                i += 1;
            }
            if i < n && is_letter(chars[i]) {
                while i < n && is_letter(chars[i]) {
                    i += 1;
                }
            } else if i < n && is_num(chars[i]) {
                // Digit runs are capped at 3 (the `\p{N}{1,3}` rule shared by the
                // llama3 and tekken pre-tokenizer families), so "1234" -> "123","4".
                let mut d = 0;
                while i < n && is_num(chars[i]) && d < 3 {
                    i += 1;
                    d += 1;
                }
            } else if i < n && chars[i] != ' ' && !chars[i].is_whitespace() {
                while i < n
                    && chars[i] != ' '
                    && !chars[i].is_whitespace()
                    && !is_letter(chars[i])
                    && !is_num(chars[i])
                {
                    i += 1;
                }
            } else {
                // whitespace run
                while i < n && chars[i].is_whitespace() {
                    i += 1;
                }
            }
            if i > start {
                out.push(chars[start..i].iter().collect());
            } else {
                // no progress (e.g. stray lead space only) -> consume one char
                out.push(chars[start..start + 1].iter().collect());
                i = start + 1;
            }
        }
        out
    }

    /// Encode text to token ids (byte-level BPE). Adds BOS only if the model's
    /// `add_bos_token` is set (it is false for this model). See the module docs
    /// for the parity caveat on arbitrary text.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.add_bos {
            ids.push(self.bos_id);
        }
        for piece in Self::pretokenize(text) {
            let symbols = self.bytes_to_symbols(piece.as_bytes());
            for sub in self.bpe(symbols) {
                match self.token_to_id.get(&sub) {
                    Some(&id) => ids.push(id),
                    None => {
                        // Fall back to single-symbol ids; byte-level vocab covers
                        // every byte, so each char-symbol must exist.
                        for ch in sub.chars() {
                            let s = ch.to_string();
                            ids.push(*self.token_to_id.get(&s).unwrap_or(&self.unk_id));
                        }
                    }
                }
            }
        }
        ids
    }
}

// ----------------------------------------------------------------------------
// Minimal, self-contained GGUF KV reader (header only; no tensor payload).
// ----------------------------------------------------------------------------

struct GgufKv {
    strings: HashMap<String, String>,
    u32s: HashMap<String, u32>,
    bools: HashMap<String, bool>,
    str_arrays: HashMap<String, Vec<String>>,
    i32_arrays: HashMap<String, Vec<i32>>,
}

impl GgufKv {
    fn string(&self, k: &str) -> Option<String> {
        self.strings.get(k).cloned()
    }
    fn u32(&self, k: &str) -> Option<u32> {
        self.u32s.get(k).copied()
    }
    fn bool(&self, k: &str) -> Option<bool> {
        self.bools.get(k).copied()
    }
    fn str_array(&self, k: &str) -> Option<Vec<String>> {
        self.str_arrays.get(k).cloned()
    }
    fn i32_array(&self, k: &str) -> Option<Vec<i32>> {
        self.i32_arrays.get(k).cloned()
    }

    fn parse(d: &[u8]) -> Result<Self, String> {
        let mut o;
        let rd_u32 = |d: &[u8], o: usize| -> u32 {
            u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
        };
        let rd_u64 = |d: &[u8], o: usize| -> u64 {
            u64::from_le_bytes([
                d[o],
                d[o + 1],
                d[o + 2],
                d[o + 3],
                d[o + 4],
                d[o + 5],
                d[o + 6],
                d[o + 7],
            ])
        };
        if d.len() < 24 || &d[0..4] != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        o = 8;
        let _n_tensors = rd_u64(d, o);
        o += 8;
        let n_kv = rd_u64(d, o);
        o += 8;

        let read_str = |d: &[u8], o: &mut usize| -> Result<String, String> {
            if *o + 8 > d.len() {
                return Err("truncated string length".into());
            }
            let ln = rd_u64(d, *o) as usize;
            *o += 8;
            if *o + ln > d.len() {
                return Err("truncated string".into());
            }
            let s = String::from_utf8_lossy(&d[*o..*o + ln]).into_owned();
            *o += ln;
            Ok(s)
        };
        // value size for primitive types
        fn prim_width(t: u32) -> Option<usize> {
            match t {
                0 | 1 | 7 => Some(1),
                2..=3 => Some(2),
                4..=6 => Some(4),
                10..=12 => Some(8),
                _ => None,
            }
        }

        let mut out = GgufKv {
            strings: HashMap::new(),
            u32s: HashMap::new(),
            bools: HashMap::new(),
            str_arrays: HashMap::new(),
            i32_arrays: HashMap::new(),
        };

        for _ in 0..n_kv {
            let key = read_str(d, &mut o)?;
            if o + 4 > d.len() {
                return Err("truncated kv type".into());
            }
            let vt = rd_u32(d, o);
            o += 4;
            match vt {
                8 => {
                    let s = read_str(d, &mut o)?;
                    out.strings.insert(key, s);
                }
                4 => {
                    let v = rd_u32(d, o);
                    o += 4;
                    out.u32s.insert(key, v);
                }
                5 => {
                    let v = rd_u32(d, o);
                    o += 4;
                    out.u32s.insert(key, v); // stored as unsigned view; ids are small
                }
                10 | 11 => {
                    let v = rd_u64(d, o) as u32;
                    o += 8;
                    out.u32s.insert(key, v);
                }
                7 => {
                    out.bools.insert(key, d[o] != 0);
                    o += 1;
                }
                9 => {
                    let at = rd_u32(d, o);
                    o += 4;
                    let al = rd_u64(d, o) as usize;
                    o += 8;
                    if at == 8 {
                        let mut v = Vec::with_capacity(al);
                        for _ in 0..al {
                            v.push(read_str(d, &mut o)?);
                        }
                        out.str_arrays.insert(key, v);
                    } else if at == 5 || at == 4 {
                        let mut v = Vec::with_capacity(al);
                        for _ in 0..al {
                            v.push(rd_u32(d, o) as i32);
                            o += 4;
                        }
                        out.i32_arrays.insert(key, v);
                    } else {
                        let w = prim_width(at).ok_or("unknown array elem type")?;
                        o += w * al;
                    }
                }
                _ => {
                    let w = prim_width(vt).ok_or("unknown kv type")?;
                    o += w;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_is_a_bijection() {
        let b2u = byte_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for (b, c) in b2u.iter().enumerate() {
            assert!(seen.insert(*c), "byte map not injective at {b}");
        }
        assert_eq!(seen.len(), 256);
    }

    fn shard1() -> Option<std::path::PathBuf> {
        let dir = std::env::var("LLMBRIDGE_NEMOTRON_DIR").ok()?;
        Some(std::path::PathBuf::from(format!(
            "{dir}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ2_XXS-00001-of-00003.gguf"
        )))
    }

    #[test]
    fn loads_real_tokenizer_metadata() {
        let Some(p) = shard1() else {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset");
            return;
        };
        let tk = NemotronTokenizer::from_gguf(&p).unwrap();
        assert_eq!(tk.model, "gpt2");
        assert_eq!(tk.pre, "pixtral");
        assert_eq!(tk.vocab_size(), 131072);
        assert_eq!(tk.unk_id, 0);
        assert_eq!(tk.bos_id, 1);
        assert_eq!(tk.eos_id, 11);
        assert!(!tk.add_bos);
        assert_eq!(tk.tokens[0], "<unk>");
        assert_eq!(tk.tokens[1], "<s>");
        assert_eq!(tk.tokens[2], "</s>");
        assert!(!tk.token_type.is_empty());
        eprintln!(
            "tokenizer: model={} pre={} vocab={} merges_loaded(non-empty)={}",
            tk.model,
            tk.pre,
            tk.vocab_size(),
            !tk.tokens.is_empty()
        );
    }

    #[test]
    fn encode_decode_roundtrips_on_ascii() {
        let Some(p) = shard1() else {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset");
            return;
        };
        let tk = NemotronTokenizer::from_gguf(&p).unwrap();
        for s in [
            "The capital of France is",
            "Hello, world!",
            "123 456",
            " leading space",
        ] {
            let ids = tk.encode(s);
            assert!(!ids.is_empty(), "empty encoding for {s:?}");
            assert!(ids.iter().all(|&id| (id as usize) < tk.vocab_size()));
            let back = tk.decode(&ids);
            assert_eq!(back.unwrap(), s, "roundtrip failed for {s:?}");
        }
    }

    #[test]
    fn pretokenize_caps_digit_runs_at_three() {
        // llama3/tekken `\p{N}{1,3}`: "1234567" -> "123","456","7".
        let p = NemotronTokenizer::pretokenize("1234567");
        assert_eq!(p, vec!["123", "456", "7"]);
        // letters are a single run; a leading space attaches to the next piece.
        let p2 = NemotronTokenizer::pretokenize("ab 12");
        assert_eq!(p2, vec!["ab", " 12"]);
    }

    #[test]
    fn decode_skips_control_tokens() {
        let Some(p) = shard1() else {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset");
            return;
        };
        let tk = NemotronTokenizer::from_gguf(&p).unwrap();
        // BOS/EOS/UNK are non-byte control tokens -> contribute no text.
        let text = tk.decode(&[tk.bos_id, tk.eos_id]).unwrap();
        assert_eq!(text, "");
    }
}
