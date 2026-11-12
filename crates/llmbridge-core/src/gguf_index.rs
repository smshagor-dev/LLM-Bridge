// Rust 1.82 MSRV predates the stable is_multiple_of integer method.
// These checked nonzero block/alignment divisibility tests intentionally use %.
#![allow(clippy::manual_is_multiple_of)]

use crate::storage::{BoundedLocalReader, SourceIdentity};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_METADATA_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TENSORS: u64 = 65_536;
const MAX_KV: u64 = 100_000;
const MAX_STRING_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_ALIGNMENT: u64 = 4096;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRegion {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    /// Offset relative to the start of the GGUF tensor-data segment.
    pub relative_offset: u64,
    /// Absolute byte offset of this tensor within the source GGUF file.
    pub file_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug)]
pub struct GgufTensorIndex {
    pub path: PathBuf,
    pub file_bytes: u64,
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub tensors: Vec<TensorRegion>,
    pub source_identity: SourceIdentity,
}

struct MetadataReader {
    file: BufReader<File>,
    offset: u64,
    file_bytes: u64,
    source_identity: SourceIdentity,
}

impl MetadataReader {
    fn open(path: &Path) -> io::Result<Self> {
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "GGUF tensor index requires a regular local file",
            ));
        }
        let file = File::open(path)?;
        let source_identity = SourceIdentity::from_metadata(&file.metadata()?);
        let file_bytes = source_identity.file_bytes;
        Ok(Self {
            file: BufReader::with_capacity(64 * 1024, file),
            offset: 0,
            file_bytes,
            source_identity,
        })
    }

    fn advance(&mut self, amount: u64) -> io::Result<u64> {
        let end = self
            .offset
            .checked_add(amount)
            .ok_or_else(|| invalid("metadata overflow"))?;
        if end > self.file_bytes || end > MAX_METADATA_BYTES {
            return Err(invalid("GGUF metadata exceeds file or bounded parse limit"));
        }
        self.offset = end;
        Ok(end)
    }

    fn read_exact<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        self.advance(N as u64)?;
        let mut bytes = [0; N];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn skip(&mut self, amount: u64) -> io::Result<()> {
        let end = self.advance(amount)?;
        self.file.seek(SeekFrom::Start(end))?;
        Ok(())
    }

    fn string(&mut self, max_bytes: u64) -> io::Result<String> {
        let count = self.u64()?;
        if count == 0 || count > max_bytes || count > MAX_STRING_BYTES {
            return Err(invalid("empty or oversized GGUF key or tensor name"));
        }
        self.advance(count)?;
        let mut bytes = vec![0u8; count as usize];
        self.file.read_exact(&mut bytes)?;
        if bytes.contains(&0) {
            return Err(invalid("NUL in GGUF key or tensor name"));
        }
        String::from_utf8(bytes).map_err(|_| invalid("GGUF key or tensor name is not UTF-8"))
    }

    fn skip_string(&mut self) -> io::Result<()> {
        let count = self.u64()?;
        if count > MAX_STRING_BYTES {
            return Err(invalid("oversized GGUF string value"));
        }
        self.skip(count)
    }

    fn skip_value(&mut self, value_type: u32) -> io::Result<()> {
        if value_type == 9 {
            // GGUF array: element type u32, element count u64, then data.
            // Recursive arrays are forbidden by the format.
            let element_type = self.u32()?;
            let count = self.u64()?;
            if count > 5_000_000 {
                return Err(invalid("GGUF metadata array too long"));
            }
            if element_type == 8 {
                for _ in 0..count {
                    self.skip_string()?;
                }
                return Ok(());
            }
            let width = primitive_width(element_type)
                .ok_or_else(|| invalid("invalid GGUF array element type"))?;
            return self.skip(
                count
                    .checked_mul(width)
                    .ok_or_else(|| invalid("array size overflow"))?,
            );
        }
        if value_type == 8 {
            return self.skip_string();
        }
        let width =
            primitive_width(value_type).ok_or_else(|| invalid("unknown GGUF metadata type"))?;
        self.skip(width)
    }
}

fn primitive_width(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Verified sizes for a narrow subset of ggml v0.9.0 tensor types.
/// Unknown layouts are rejected, not guessed or reinterpreted.
fn tensor_layout(ggml_type: u32) -> Option<(u64, u64)> {
    match ggml_type {
        0 => Some((1, 4)),      // F32
        1 => Some((1, 2)),      // F16
        2 => Some((32, 18)),    // Q4_0
        3 => Some((32, 20)),    // Q4_1
        6 => Some((32, 22)),    // Q5_0
        7 => Some((32, 24)),    // Q5_1
        8 => Some((32, 34)),    // Q8_0
        9 => Some((32, 40)),    // Q8_1
        10 => Some((256, 84)),  // Q2_K
        11 => Some((256, 110)), // Q3_K
        12 => Some((256, 144)), // Q4_K
        13 => Some((256, 176)), // Q5_K
        14 => Some((256, 210)), // Q6_K
        15 => Some((256, 292)), // Q8_K
        30 => Some((1, 2)),     // BF16
        _ => None,
    }
}

fn align_up(offset: u64, alignment: u64) -> io::Result<u64> {
    offset
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| invalid("aligned metadata offset overflow"))
}

impl GgufTensorIndex {
    /// Read GGUF v2/v3 metadata and index tensor byte ranges without loading tensor payloads.
    /// Bounded metadata parsing prevents untrusted declared lengths from allocating whole weights.
    /// A tensor whose quantization layout is not explicitly known is rejected.
    pub fn inspect(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let mut reader = MetadataReader::open(path)?;
        if reader.read_exact::<4>()? != *b"GGUF" {
            return Err(invalid("not a GGUF file"));
        }
        let version = reader.u32()?;
        if version != 2 && version != 3 {
            return Err(invalid("only GGUF v2/v3 tensor indexing supported"));
        }
        let tensor_count = reader.u64()?;
        let kv_count = reader.u64()?;
        if tensor_count == 0 || tensor_count > MAX_TENSORS || kv_count > MAX_KV {
            return Err(invalid(
                "GGUF tensor or metadata count exceeds index admission",
            ));
        }

        let mut alignment = DEFAULT_ALIGNMENT;
        let mut seen_alignment = false;
        let mut seen_keys = HashSet::new();
        for _ in 0..kv_count {
            let key = reader.string(256)?;
            if !seen_keys.insert(key.clone()) {
                return Err(invalid("duplicate GGUF metadata key"));
            }
            let value_type = reader.u32()?;
            if key == "general.alignment" {
                if seen_alignment || value_type != 4 {
                    return Err(invalid("GGUF general.alignment must be a single uint32"));
                }
                seen_alignment = true;
                alignment = u64::from(reader.u32()?);
            } else {
                reader.skip_value(value_type)?;
            }
        }
        if !alignment.is_power_of_two() || alignment > MAX_ALIGNMENT {
            return Err(invalid("GGUF alignment must be a supported power of two"));
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut seen_tensors = HashSet::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.string(127)?;
            if !seen_tensors.insert(name.clone()) {
                return Err(invalid("duplicate GGUF tensor name"));
            }
            let dimension_count = reader.u32()?;
            if !(1..=4).contains(&dimension_count) {
                return Err(invalid("GGUF tensor rank must be 1..4"));
            }
            let mut dimensions = Vec::with_capacity(dimension_count as usize);
            let mut elements = 1u64;
            for _ in 0..dimension_count {
                let dimension = reader.u64()?;
                if dimension == 0 {
                    return Err(invalid(
                        "zero-sized tensors are not supported by this reader",
                    ));
                }
                elements = elements
                    .checked_mul(dimension)
                    .ok_or_else(|| invalid("GGUF tensor element count overflow"))?;
                dimensions.push(dimension);
            }
            let ggml_type = reader.u32()?;
            let (block, bytes_per_block) = tensor_layout(ggml_type)
                .ok_or_else(|| invalid("GGUF tensor type not supported for indexed range reads"))?;
            if dimensions[0] % block != 0 || elements % block != 0 {
                return Err(invalid(
                    "GGUF tensor dimension violates quantization block layout",
                ));
            }
            let byte_len = (elements / block)
                .checked_mul(bytes_per_block)
                .ok_or_else(|| invalid("GGUF tensor byte size overflow"))?;
            let relative_offset = reader.u64()?;
            if relative_offset % alignment != 0 {
                return Err(invalid("misaligned GGUF tensor offset"));
            }
            tensors.push(TensorRegion {
                name,
                dimensions,
                ggml_type,
                relative_offset,
                file_offset: 0,
                byte_len,
            });
        }
        let data_offset = align_up(reader.offset, alignment)?;
        if data_offset > reader.file_bytes {
            return Err(invalid("GGUF tensor data begins beyond end of file"));
        }
        let mut ranges = Vec::with_capacity(tensors.len());
        for t in &mut tensors {
            t.file_offset = data_offset
                .checked_add(t.relative_offset)
                .ok_or_else(|| invalid("GGUF tensor absolute offset overflow"))?;
            let end = t
                .file_offset
                .checked_add(t.byte_len)
                .ok_or_else(|| invalid("GGUF tensor range overflow"))?;
            if end > reader.file_bytes {
                return Err(invalid("GGUF tensor range exceeds file length"));
            }
            ranges.push((t.file_offset, end));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(invalid("GGUF tensor byte regions overlap"));
            }
        }

        if SourceIdentity::from_metadata(&reader.file.get_ref().metadata()?) != reader.source_identity {
            return Err(invalid("GGUF file changed during tensor index construction"));
        }
        Ok(Self {
            path: path.to_path_buf(),
            file_bytes: reader.file_bytes,
            source_identity: reader.source_identity,
            version,
            alignment,
            data_offset,
            tensors,
        })
    }

    /// Hashes exactly the indexed tensor payload, excluding padding and other weights.
    pub fn sha256_tensor(&self, reader: &mut BoundedLocalReader, name: &str) -> io::Result<String> {
        let mut hash = Sha256::new();
        self.stream_tensor(reader, name, |chunk| {
            hash.update(chunk);
            Ok(())
        })?;
        Ok(format!("{:x}", hash.finalize()))
    }

    /// Rejects streamed tensor data that fails its caller-supplied integrity digest.
    pub fn verify_tensor_sha256(
        &self,
        reader: &mut BoundedLocalReader,
        name: &str,
        expected: &str,
    ) -> io::Result<String> {
        if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tensor digest must be 64 hex digits",
            ));
        }
        let actual = self.sha256_tensor(reader, name)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(invalid("GGUF tensor SHA-256 mismatch"));
        }
        Ok(actual)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorRegion> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Streams exact declared tensor bytes through the bounded local reader.
    /// Does not interpret numeric values, execute a model, or certify their provenance.
    pub fn stream_tensor<F>(
        &self,
        reader: &mut BoundedLocalReader,
        name: &str,
        mut consume: F,
    ) -> io::Result<u64>
    where
        F: FnMut(&[u8]) -> io::Result<()>,
    {

        if reader.identity() != &self.source_identity {
            return Err(invalid("indexed tensor source identity changed before streaming"));
        }
        let tensor = self.tensor(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "tensor not found in indexed GGUF")
        })?;
        let mut offset = tensor.file_offset;
        let end = offset
            .checked_add(tensor.byte_len)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        while offset < end {
            let count = (end - offset).min(reader.allocated_chunk_bytes() as u64) as usize;
            let chunk = reader.read_range(offset, count)?;
            consume(chunk)?;
            offset += count as u64;
        }
        Ok(tensor.byte_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);

    fn str_field(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn fixture(qtype: u32, first_relative_offset: u64, second_relative_offset: u64) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        str_field(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // uint32
        bytes.extend_from_slice(&32u32.to_le_bytes());

        str_field(&mut bytes, "layers.0.weight");
        bytes.extend_from_slice(&2u32.to_le_bytes()); // dimensions
        bytes.extend_from_slice(&32u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&qtype.to_le_bytes());
        bytes.extend_from_slice(&first_relative_offset.to_le_bytes());

        str_field(&mut bytes, "final.weight");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // F16
        bytes.extend_from_slice(&second_relative_offset.to_le_bytes());

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0xA5; 68]); // 2 x Q8_0 blocks
        bytes.extend_from_slice(&[0; 28]); // padding
        bytes.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]); // final F16 values
        bytes
    }

    fn write(bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "llmbridge-index-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }

    #[test]
    fn indexed_tensor_ranges_are_exact_and_bounded() {
        let path = write(&fixture(8, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        assert_eq!(index.tensors.len(), 2);
        let weight = index.tensor("layers.0.weight").unwrap();
        assert_eq!(
            (
                weight.dimensions.as_slice(),
                weight.ggml_type,
                weight.byte_len
            ),
            (&[32, 2][..], 8, 68)
        );
        assert_eq!(weight.file_offset % 32, 0);
        let mut reader = BoundedLocalReader::open(&path, 13, 26).unwrap();
        let mut output = Vec::new();
        index
            .stream_tensor(&mut reader, &weight.name, |chunk| {
                assert!(chunk.len() <= 13);
                output.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(output, vec![0xA5; 68]);
        assert_eq!(reader.disk_bytes_read(), 68);
        assert!(index
            .verify_tensor_sha256(&mut reader, &weight.name, &"0".repeat(64))
            .is_err());
        assert_eq!(reader.allocated_chunk_bytes(), 13);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mismatched_hash_or_region_is_not_silent() {
        let path = write(&fixture(8, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 16, 32).unwrap();
        let mut hasher = Sha256::new();
        index
            .stream_tensor(&mut reader, "final.weight", |b| {
                hasher.update(b);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            format!("{:x}", hasher.finalize()),
            format!("{:x}", Sha256::digest([0x12, 0x34, 0x56, 0x78]))
        );
        assert!(index
            .stream_tensor(&mut reader, "missing", |_| Ok(()))
            .is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_tensor_layout_and_ranges_are_rejected() {
        for (kind, first, second) in [(999, 0, 96), (8, 1, 96), (8, 0, 0), (8, 0, 4096)] {
            let path = write(&fixture(kind, first, second));
            assert!(GgufTensorIndex::inspect(&path).is_err());
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn truncated_input_does_not_allocate_weights() {
        let mut bytes = fixture(8, 0, 96);
        bytes.truncate(bytes.len() - 2);
        let path = write(&bytes);
        assert!(GgufTensorIndex::inspect(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }


    #[cfg(unix)]
    #[test]
    fn reject_same_size_source_replacement_with_new_inode() {
        let path = write(&fixture(8, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let replacement = write(&fixture(8, 0, 96));
        std::fs::rename(&replacement, &path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 16, 32).unwrap();
        assert!(index.stream_tensor(&mut reader, "final.weight", |_| Ok(())).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn data_offset_and_alignment_are_enforced() {
        assert_eq!(align_up(65, 32).unwrap(), 96);
        assert!(align_up(u64::MAX, 32).is_err());
    }
}
