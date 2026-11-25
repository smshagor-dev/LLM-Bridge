// Rust 1.82 MSRV predates the stable is_multiple_of integer method.
// These checked nonzero block/alignment divisibility tests intentionally use %.

use crate::storage::{BoundedLocalReader, SourceIdentity};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

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

struct MetadataReader<F> {
    file: BufReader<File>,
    offset: u64,
    file_bytes: u64,
    source_identity: SourceIdentity,
    interrupted: F,
}

impl<F: Fn() -> bool> MetadataReader<F> {
    fn open(path: &Path, interrupted: F) -> io::Result<Self> {
        check_inspection(interrupted())?;
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "GGUF tensor index requires a regular local file",
            ));
        }
        let file = File::open(path)?;
        let opened = file.metadata()?;
        let source_identity = SourceIdentity::from_file(&file)?;
        let path_file = File::open(path)?;
        // The path can be replaced between symlink_metadata and File::open.
        // Check both the opened descriptor and the current path before parsing.
        if !opened.is_file()
            || source_identity != SourceIdentity::from_file(&path_file)?
            || std::fs::symlink_metadata(path)
                .map(|current| {
                    current.file_type().is_symlink()
                        || !current.is_file()
                        || File::open(path)
                            .and_then(|current| SourceIdentity::from_file(&current))
                            .map(|identity| identity != source_identity)
                            .unwrap_or(true)
                })
                .unwrap_or(true)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "GGUF metadata source changed during open",
            ));
        }
        let file_bytes = source_identity.file_bytes;
        Ok(Self {
            file: BufReader::with_capacity(64 * 1024, file),
            offset: 0,
            file_bytes,
            source_identity,
            interrupted,
        })
    }

    fn advance(&mut self, amount: u64) -> io::Result<()> {
        check_inspection((self.interrupted)())?;
        let end = self
            .offset
            .checked_add(amount)
            .ok_or_else(|| invalid("metadata overflow"))?;
        if end > self.file_bytes || end > MAX_METADATA_BYTES {
            return Err(invalid("GGUF metadata exceeds file or bounded parse limit"));
        }
        self.offset = end;
        Ok(())
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

    /// Read a GGUF integer scalar of the given value type as u64. Split metadata
    /// keys (split.no/count are uint16, split.tensors.count is int32 as written by
    /// gguf-split) must be genuine non-negative integers; anything else is rejected
    /// rather than coerced.
    fn scalar_u64(&mut self, value_type: u32) -> io::Result<u64> {
        match value_type {
            2 => Ok(u64::from(u16::from_le_bytes(self.read_exact()?))),
            3 => {
                let v = i16::from_le_bytes(self.read_exact()?);
                u64::try_from(v).map_err(|_| invalid("negative GGUF split metadata value"))
            }
            4 => Ok(u64::from(self.u32()?)),
            5 => {
                let v = i32::from_le_bytes(self.read_exact()?);
                u64::try_from(v).map_err(|_| invalid("negative GGUF split metadata value"))
            }
            10 => self.u64(),
            11 => {
                let v = i64::from_le_bytes(self.read_exact()?);
                u64::try_from(v).map_err(|_| invalid("negative GGUF split metadata value"))
            }
            _ => Err(invalid("GGUF split metadata value is not an integer")),
        }
    }

    fn skip(&mut self, amount: u64) -> io::Result<()> {
        self.advance(amount)?;
        // The bounded metadata size fits in i64. Keep buffered bytes when the
        // skipped value (often a short tokenizer string) fits in the buffer.
        self.file.seek_relative(amount as i64)
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

fn check_inspection(interrupted: bool) -> io::Result<()> {
    if interrupted {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "GGUF metadata inspection interrupted",
        ));
    }
    Ok(())
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
        // i-quant block geometries (indexing/range-read only; NOT a dequant kernel).
        // Byte sizes are the pinned ggml type_size constants: a value that is a
        // verifiable layout fact, independent of any decode implementation.
        16 => Some((256, 66)), // IQ2_XXS: {half d; u16 qs[QK_K/8]} = 2 + 32*2
        20 => Some((32, 18)),  // IQ4_NL:  {half d; u8 qs[QK4_NL/2]} = 2 + 16
        30 => Some((1, 2)),    // BF16
        _ => None,
    }
}

fn align_up(offset: u64, alignment: u64) -> io::Result<u64> {
    offset
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| invalid("aligned metadata offset overflow"))
}

/// Validate both the descriptor and the pathname: checking only the open
/// descriptor misses rename/unlink replacement during metadata parsing.
fn assert_metadata_source_unchanged(
    path: &Path,
    file: &File,
    expected: &SourceIdentity,
) -> io::Result<()> {
    let current_path = std::fs::symlink_metadata(path)?;
    let current_file = File::open(path)?;
    if current_path.file_type().is_symlink()
        || !current_path.is_file()
        || SourceIdentity::from_file(&current_file)? != *expected
        || SourceIdentity::from_file(file)? != *expected
    {
        return Err(invalid(
            "GGUF file or path changed during tensor index construction",
        ));
    }
    Ok(())
}

impl GgufTensorIndex {
    /// Read GGUF v2/v3 metadata and index tensor byte ranges without loading tensor payloads.
    /// Bounded metadata parsing prevents untrusted declared lengths from allocating whole weights.
    /// A tensor whose quantization layout is not explicitly known is rejected.
    pub fn inspect(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::inspect_cancellable(path.as_ref(), || false)
    }

    pub(crate) fn inspect_cancellable(
        path: &Path,
        interrupted: impl Fn() -> bool,
    ) -> io::Result<Self> {
        let mut reader = MetadataReader::open(path, interrupted)?;
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
            check_inspection((reader.interrupted)())?;
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

        assert_metadata_source_unchanged(path, reader.file.get_ref(), &reader.source_identity)?;
        check_inspection((reader.interrupted)())?;
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

    /// Hash only the indexed tensor region with cooperative cancellation.
    /// The digest is returned only after every byte has been consumed; on
    /// cancellation, the partial hash is discarded. The expected digest must
    /// still be authenticated independently by the caller.
    pub fn sha256_tensor_cancellable(
        &self,
        reader: &mut BoundedLocalReader,
        name: &str,
        cancelled: &AtomicBool,
    ) -> io::Result<String> {
        let mut hash = Sha256::new();
        self.stream_tensor_cancellable(reader, name, cancelled, |chunk| {
            hash.update(chunk);
            Ok(())
        })?;
        Ok(format!("{:x}", hash.finalize()))
    }

    /// Compare an indexed tensor against a trusted digest, aborting promptly
    /// between bounded reads. No digest is returned for partial input.
    pub fn verify_tensor_sha256_cancellable(
        &self,
        reader: &mut BoundedLocalReader,
        name: &str,
        expected: &str,
        cancelled: &AtomicBool,
    ) -> io::Result<String> {
        if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tensor digest must be 64 hex digits",
            ));
        }
        let actual = self.sha256_tensor_cancellable(reader, name, cancelled)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(invalid("GGUF tensor SHA-256 mismatch"));
        }
        Ok(actual)
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
        consume: F,
    ) -> io::Result<u64>
    where
        F: FnMut(&[u8]) -> io::Result<()>,
    {
        self.stream_tensor_cancellable(reader, name, &AtomicBool::new(false), consume)
    }

    /// Stream an exact subrange of an indexed tensor without materializing it.
    /// Offsets are relative to the tensor, not the GGUF file. All bounds are
    /// checked before any I/O; cancellation can leave partial consumer output.
    pub fn stream_tensor_slice_cancellable<F>(
        &self,
        reader: &mut BoundedLocalReader,
        name: &str,
        relative_offset: u64,
        length: u64,
        cancelled: &AtomicBool,
        mut consume: F,
    ) -> io::Result<u64>
    where
        F: FnMut(&[u8]) -> io::Result<()>,
    {
        if reader.identity() != &self.source_identity {
            return Err(invalid(
                "indexed tensor source identity changed before streaming",
            ));
        }
        let tensor = self.tensor(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "tensor not found in indexed GGUF")
        })?;
        let relative_end = relative_offset
            .checked_add(length)
            .ok_or_else(|| invalid("tensor slice length overflow"))?;
        if relative_end > tensor.byte_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tensor slice exceeds indexed region",
            ));
        }
        let mut offset = tensor
            .file_offset
            .checked_add(relative_offset)
            .ok_or_else(|| invalid("tensor slice file offset overflow"))?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| invalid("tensor slice file end overflow"))?;
        loop {
            if cancelled.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "tensor stream cancelled",
                ));
            }
            if offset == end {
                return Ok(length);
            }
            let count = (end - offset).min(reader.allocated_chunk_bytes() as u64) as usize;
            let chunk = reader.read_range(offset, count)?;
            consume(chunk)?;
            offset += count as u64;
        }
    }

    /// Stream a complete indexed tensor using the same bounded slice path.
    pub fn stream_tensor_cancellable<F>(
        &self,
        reader: &mut BoundedLocalReader,
        name: &str,
        cancelled: &AtomicBool,
        consume: F,
    ) -> io::Result<u64>
    where
        F: FnMut(&[u8]) -> io::Result<()>,
    {
        let tensor = self.tensor(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "tensor not found in indexed GGUF")
        })?;
        self.stream_tensor_slice_cancellable(reader, name, 0, tensor.byte_len, cancelled, consume)
    }
}

/// Split ("sharded") GGUF metadata read from one shard: the self-describing
/// `split.*` keys plus the shard's own tensor count. Values come from metadata,
/// never from the filename, so a renamed or reordered shard cannot masquerade as
/// a different member.
#[derive(Debug, Clone)]
struct SplitKeys {
    tensor_count: u64,
    split_no: Option<u64>,
    split_count: Option<u64>,
    split_tensors_count: Option<u64>,
}

/// Parse only the GGUF header and metadata key/value block of a shard, capturing
/// the `split.*` keys. Unlike `GgufTensorIndex::inspect`, this permits a
/// zero-tensor header shard (the metadata-only split member) and does not parse
/// the tensor table. Bounded exactly like the full indexer.
fn read_split_keys(path: &Path, interrupted: impl Fn() -> bool) -> io::Result<SplitKeys> {
    let mut reader = MetadataReader::open(path, interrupted)?;
    if reader.read_exact::<4>()? != *b"GGUF" {
        return Err(invalid("not a GGUF file"));
    }
    let version = reader.u32()?;
    if version != 2 && version != 3 {
        return Err(invalid("only GGUF v2/v3 split indexing supported"));
    }
    let tensor_count = reader.u64()?;
    let kv_count = reader.u64()?;
    if tensor_count > MAX_TENSORS || kv_count > MAX_KV {
        return Err(invalid(
            "GGUF tensor or metadata count exceeds split admission",
        ));
    }
    let (mut split_no, mut split_count, mut split_tensors_count) = (None, None, None);
    let mut seen_keys = HashSet::new();
    for _ in 0..kv_count {
        let key = reader.string(256)?;
        if !seen_keys.insert(key.clone()) {
            return Err(invalid("duplicate GGUF metadata key"));
        }
        let value_type = reader.u32()?;
        match key.as_str() {
            "split.no" => split_no = Some(reader.scalar_u64(value_type)?),
            "split.count" => split_count = Some(reader.scalar_u64(value_type)?),
            "split.tensors.count" => split_tensors_count = Some(reader.scalar_u64(value_type)?),
            _ => reader.skip_value(value_type)?,
        }
    }
    Ok(SplitKeys {
        tensor_count,
        split_no,
        split_count,
        split_tensors_count,
    })
}

/// One member of a multi-shard (split) GGUF model.
#[derive(Debug)]
pub struct SplitMember {
    pub path: PathBuf,
    /// 0-based position declared by this shard's `split.no` metadata.
    pub split_no: u64,
    /// Tensor count from this shard's own GGUF header.
    pub tensor_count: u64,
    /// Full validated tensor index for a data shard; `None` for a zero-tensor
    /// metadata header shard.
    pub index: Option<GgufTensorIndex>,
}

/// A logical index over a set of split GGUF shards. Tensor identity, partitioning
/// and split completeness are validated from `split.*` metadata and each shard's
/// own tensor table — the shards are never concatenated, and each retains an
/// independent source identity and byte range for bounded, per-shard range reads.
#[derive(Debug)]
pub struct SplitGgufIndex {
    pub split_count: u64,
    pub total_tensors: u64,
    /// Members sorted ascending by `split_no`.
    pub members: Vec<SplitMember>,
    tensor_owner: HashMap<String, usize>,
}

impl SplitGgufIndex {
    /// Validate and index a complete set of split shards. The set is admitted only
    /// when every shard agrees on `split.count` and `split.tensors.count`, the
    /// `split.no` values are exactly `0..split_count` with no gaps or duplicates,
    /// the number of shards equals `split.count`, no tensor name appears in more
    /// than one shard, and the aggregate indexed tensor count equals
    /// `split.tensors.count`. No tensor payload is read.
    pub fn inspect<P: AsRef<Path>>(paths: &[P]) -> io::Result<Self> {
        Self::inspect_cancellable(paths, || false)
    }

    pub fn inspect_cancellable<P: AsRef<Path>>(
        paths: &[P],
        interrupted: impl Fn() -> bool,
    ) -> io::Result<Self> {
        if paths.is_empty() {
            return Err(invalid("split GGUF model requires at least one shard"));
        }
        if paths.len() > MAX_TENSORS as usize {
            return Err(invalid("split GGUF shard count exceeds admission"));
        }
        let mut members = Vec::with_capacity(paths.len());
        let mut split_count_global: Option<u64> = None;
        let mut total_global: Option<u64> = None;
        for path in paths {
            let path = path.as_ref();
            let keys = read_split_keys(path, &interrupted)?;
            let split_no = keys
                .split_no
                .ok_or_else(|| invalid("shard missing split.no metadata"))?;
            let split_count = keys
                .split_count
                .ok_or_else(|| invalid("shard missing split.count metadata"))?;
            let total = keys
                .split_tensors_count
                .ok_or_else(|| invalid("shard missing split.tensors.count metadata"))?;
            if split_count == 0 {
                return Err(invalid("split.count must be non-zero"));
            }
            match split_count_global {
                None => split_count_global = Some(split_count),
                Some(prior) if prior != split_count => {
                    return Err(invalid("inconsistent split.count across shards"));
                }
                _ => {}
            }
            match total_global {
                None => total_global = Some(total),
                Some(prior) if prior != total => {
                    return Err(invalid("inconsistent split.tensors.count across shards"));
                }
                _ => {}
            }
            // A data shard is fully indexed (and its tensor geometry validated) by
            // the audited single-file indexer; a zero-tensor header shard is not.
            let index = if keys.tensor_count == 0 {
                None
            } else {
                Some(GgufTensorIndex::inspect_cancellable(path, &interrupted)?)
            };
            members.push(SplitMember {
                path: path.to_path_buf(),
                split_no,
                tensor_count: keys.tensor_count,
                index,
            });
        }
        let split_count = split_count_global.expect("non-empty shard set");
        let total_tensors = total_global.expect("non-empty shard set");
        if split_count != paths.len() as u64 {
            return Err(invalid(
                "split.count does not match the number of shards provided",
            ));
        }
        members.sort_by_key(|member| member.split_no);
        for (expected, member) in members.iter().enumerate() {
            if member.split_no != expected as u64 {
                return Err(invalid("split.no sequence has a gap or duplicate"));
            }
        }
        let mut tensor_owner = HashMap::new();
        let mut aggregate = 0u64;
        for (member_index, member) in members.iter().enumerate() {
            if let Some(index) = &member.index {
                if index.tensors.len() as u64 != member.tensor_count {
                    return Err(invalid(
                        "indexed tensor count disagrees with shard header count",
                    ));
                }
                for tensor in &index.tensors {
                    if tensor_owner
                        .insert(tensor.name.clone(), member_index)
                        .is_some()
                    {
                        return Err(invalid("duplicate tensor name across split shards"));
                    }
                }
                aggregate = aggregate
                    .checked_add(index.tensors.len() as u64)
                    .ok_or_else(|| invalid("aggregate tensor count overflow"))?;
            }
        }
        if aggregate != total_tensors {
            return Err(invalid(
                "aggregate indexed tensor count disagrees with split.tensors.count",
            ));
        }
        Ok(Self {
            split_count,
            total_tensors,
            members,
            tensor_owner,
        })
    }

    /// Number of tensors indexed across every shard (equals `total_tensors`).
    pub fn indexed_tensor_count(&self) -> usize {
        self.tensor_owner.len()
    }

    /// Resolve a tensor name to the owning shard member and its byte region,
    /// mapping the logical model tensor to the correct shard and file offset
    /// without combining shards.
    pub fn tensor(&self, name: &str) -> Option<(&SplitMember, &TensorRegion)> {
        let member_index = *self.tensor_owner.get(name)?;
        let member = &self.members[member_index];
        let region = member.index.as_ref()?.tensor(name)?;
        Some((member, region))
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
    fn inspection_stops_between_metadata_fields_and_reopens_cleanly() {
        let path = write(&fixture(8, 0, 96));
        let checks = std::cell::Cell::new(0);
        let result = GgufTensorIndex::inspect_cancellable(&path, || {
            checks.set(checks.get() + 1);
            checks.get() >= 8
        });
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Interrupted);
        assert_eq!(checks.get(), 8);
        assert_eq!(GgufTensorIndex::inspect(&path).unwrap().tensors.len(), 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn skip_preserves_metadata_position_across_buffer_boundaries() {
        let mut bytes = 8u32.to_le_bytes().to_vec(); // string element type
        bytes.extend_from_slice(&3u64.to_le_bytes());
        for value in ["a", &"x".repeat(70 * 1024), ""] {
            str_field(&mut bytes, value);
        }
        bytes.extend_from_slice(&42u32.to_le_bytes());
        let path = write(&bytes);
        let mut reader = MetadataReader::open(&path, || false).unwrap();
        reader.skip_value(9).unwrap();
        assert_eq!(reader.u32().unwrap(), 42);
        assert_eq!(reader.offset, bytes.len() as u64);
        drop(reader);
        std::fs::remove_file(path).unwrap();
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
    fn q4_1_tensor_geometry_is_indexed_exactly() {
        // Q4_1 (ggml type 3) uses 32-element blocks of 20 bytes, exactly like the
        // already-supported Q4_0/Q8_0 block cadence. A [32, 2] tensor is 2 blocks =
        // 40 bytes. This pins the indexer side of the Q4_1 streamed-decoder support.
        let path = write(&fixture(3, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let weight = index.tensor("layers.0.weight").unwrap();
        assert_eq!(
            (
                weight.dimensions.as_slice(),
                weight.ggml_type,
                weight.byte_len
            ),
            (&[32, 2][..], 3, 40)
        );
        assert_eq!(weight.file_offset % 32, 0);
        let mut reader = BoundedLocalReader::open(&path, 16, 32).unwrap();
        let mut output = Vec::new();
        index
            .stream_tensor(&mut reader, &weight.name, |chunk| {
                assert!(chunk.len() <= 16);
                output.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(output, vec![0xA5; 40]);
        assert_eq!(reader.disk_bytes_read(), 40);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn kquant_and_extended_block_geometry_is_exact() {
        // Block cadence (elements) and quantized block byte size for every type
        // the streamed decoder now admits, matching the pinned ggml definitions.
        assert_eq!(tensor_layout(6), Some((32, 22))); // Q5_0
        assert_eq!(tensor_layout(7), Some((32, 24))); // Q5_1
        assert_eq!(tensor_layout(12), Some((256, 144))); // Q4_K
        assert_eq!(tensor_layout(13), Some((256, 176))); // Q5_K
        assert_eq!(tensor_layout(14), Some((256, 210))); // Q6_K
                                                         // A 768-wide Q6_K row (the stories ffn_down width) is 3 superblocks.
        let (block, bytes_per_block) = tensor_layout(14).unwrap();
        assert_eq!(768 % block, 0);
        assert_eq!((768 / block) * bytes_per_block, 630);
        // A 288-wide tensor is NOT a whole number of 256-element superblocks and
        // is rejected by the indexer's dimension-divisibility guard.
        assert_ne!(288 % block, 0);
    }

    #[test]
    fn qwen2_tensor_and_bias_geometry_is_exact() {
        // Qwen2.5-0.5B real shapes: Q8_0 (type 8) weights and F32 (type 0) Q/K/V
        // biases. dim=896, kvdim=128 (2 kv heads x 64), n_ff=4864. This pins the
        // indexer geometry the streamed qwen2 decoder depends on.
        let (q8_block, q8_bytes) = tensor_layout(8).unwrap(); // Q8_0
        assert_eq!((q8_block, q8_bytes), (32, 34));
        // Attention/embedding rows are 896 wide: a multiple of 32 but NOT of 256,
        // which is why a Q4_K_M quantization keeps these tensors non-K-quant.
        assert_eq!(896 % q8_block, 0);
        assert_ne!(896 % 256, 0);
        assert_eq!((896 / q8_block) * q8_bytes, 952); // one attn_q/attn_output row
        assert_eq!((4864 / q8_block) * q8_bytes, 5168); // one ffn_down row (4864 wide)
        assert_eq!(4864 % 256, 0); // n_ff is K-quant eligible
                                   // F32 biases are dense 4-byte rows; attn_q.bias is 896 wide, attn_k/v 128.
        let (f32_block, f32_bytes) = tensor_layout(0).unwrap();
        assert_eq!((f32_block, f32_bytes), (1, 4));
        assert_eq!(896 * f32_bytes, 3584); // attn_q.bias byte length
        assert_eq!(128 * f32_bytes, 512); // attn_k.bias / attn_v.bias byte length
    }

    #[test]
    fn llama3_rope_freqs_and_kquant_geometry_is_exact() {
        // Llama-3.2-1B real shapes: dim=2048, head_dim=64 (kvdim=512, 8 kv heads),
        // n_ff=8192, Q4_K/Q6_K weights, and the baked rope_freqs.weight [32] F32
        // (head_dim/2) per-pair frequency factors the streamed llama-3 path reads.
        let (f32_block, f32_bytes) = tensor_layout(0).unwrap();
        assert_eq!((f32_block, f32_bytes), (1, 4));
        assert_eq!(32 * f32_bytes, 128); // rope_freqs.weight byte length (head_dim/2)
                                         // All Llama-3.2 weight rows are 256-divisible, so every weight tensor is a
                                         // genuine K-quant superblock (no 32-block fallback).
        let (q4k_block, q4k_bytes) = tensor_layout(12).unwrap(); // Q4_K
        assert_eq!((q4k_block, q4k_bytes), (256, 144));
        for cols in [2048u64, 512, 8192] {
            assert_eq!(cols % q4k_block, 0);
        }
        assert_eq!((2048 / q4k_block) * q4k_bytes, 1152); // one attn_q/attn_output row
        let (q6k_block, q6k_bytes) = tensor_layout(14).unwrap(); // Q6_K
        assert_eq!((8192 / q6k_block) * q6k_bytes, 6720); // one ffn_down row (8192 wide)
    }

    #[test]
    fn indexed_tensor_slice_reads_exact_range() {
        let path = write(&fixture(8, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 3, 6).unwrap();
        let stop = AtomicBool::new(false);
        let mut output = Vec::new();
        let read = index
            .stream_tensor_slice_cancellable(&mut reader, "final.weight", 1, 2, &stop, |chunk| {
                output.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(read, 2);
        assert_eq!(output, [0x34, 0x56]);
        assert_eq!(reader.disk_bytes_read(), 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn indexed_tensor_slice_rejects_out_of_bounds_without_io() {
        let path = write(&fixture(8, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 3, 6).unwrap();
        let stop = AtomicBool::new(false);
        for (offset, length) in [(3, 2), (u64::MAX, 2), (0, 5)] {
            assert!(index
                .stream_tensor_slice_cancellable(
                    &mut reader,
                    "final.weight",
                    offset,
                    length,
                    &stop,
                    |_| Ok(())
                )
                .is_err());
        }
        assert_eq!(reader.disk_bytes_read(), 0);
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
        assert!(index
            .stream_tensor(&mut reader, "final.weight", |_| Ok(()))
            .is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn reject_replaced_path_at_metadata_completion() {
        let path = write(&fixture(8, 0, 96));
        let old_descriptor = File::open(&path).unwrap();
        let old_identity = SourceIdentity::from_metadata(&old_descriptor.metadata().unwrap());
        let replacement = write(&fixture(8, 0, 96));
        std::fs::rename(&replacement, &path).unwrap();
        let new_file = std::fs::symlink_metadata(&path).unwrap();
        assert_ne!(old_identity, SourceIdentity::from_metadata(&new_file));
        assert!(assert_metadata_source_unchanged(&path, &old_descriptor, &old_identity).is_err());
        let index = GgufTensorIndex::inspect(&path).unwrap();
        assert_eq!(
            index.source_identity,
            SourceIdentity::from_metadata(&new_file)
        );
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn reject_symlink_substitution_at_metadata_completion() {
        use std::os::unix::fs::symlink;
        let path = write(&fixture(8, 0, 96));
        let old_descriptor = File::open(&path).unwrap();
        let identity = SourceIdentity::from_metadata(&old_descriptor.metadata().unwrap());
        let other = write(&fixture(8, 0, 96));
        std::fs::remove_file(&path).unwrap();
        symlink(&other, &path).unwrap();
        assert!(assert_metadata_source_unchanged(&path, &old_descriptor, &identity).is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(other).unwrap();
    }

    #[test]
    fn cancellation_prevents_io_and_stops_between_chunks() {
        let path = write(&fixture(8, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 8, 16).unwrap();
        let stop = AtomicBool::new(true);
        let err = index
            .stream_tensor_cancellable(&mut reader, "layers.0.weight", &stop, |_| Ok(()))
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
        assert_eq!(reader.disk_bytes_read(), 0);
        stop.store(false, Ordering::Release);
        let mut delivered = 0;
        let err = index
            .stream_tensor_cancellable(&mut reader, "layers.0.weight", &stop, |chunk| {
                delivered += chunk.len();
                stop.store(true, Ordering::Release);
                Ok(())
            })
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
        assert_eq!(delivered, 8);
        assert_eq!(reader.disk_bytes_read(), 8);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn tensor_digest_cancellation_is_bounded_and_never_returns_partial_hash() {
        let path = write(&fixture(8, 0, 96));
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 8, 16).unwrap();
        let stop = AtomicBool::new(true);
        let expected = format!("{:x}", Sha256::digest([0xA5; 68]));
        let error = index
            .verify_tensor_sha256_cancellable(&mut reader, "layers.0.weight", &expected, &stop)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert_eq!(reader.disk_bytes_read(), 0);
        stop.store(false, Ordering::Release);
        assert_eq!(
            index
                .verify_tensor_sha256_cancellable(&mut reader, "layers.0.weight", &expected, &stop,)
                .unwrap(),
            expected
        );
        assert_eq!(reader.disk_bytes_read(), 68);
        assert!(index
            .verify_tensor_sha256_cancellable(
                &mut reader,
                "layers.0.weight",
                &"0".repeat(64),
                &stop,
            )
            .is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn data_offset_and_alignment_are_enforced() {
        assert_eq!(align_up(65, 32).unwrap(), 96);
        assert!(align_up(u64::MAX, 32).is_err());
    }

    #[test]
    fn iquant_block_geometry_is_exact() {
        // Pinned ggml type_size constants for the i-quants present in the Nemotron
        // UD-IQ2_XXS shards. Indexing/range-read geometry only; not a dequantizer.
        assert_eq!(tensor_layout(16), Some((256, 66))); // IQ2_XXS
        assert_eq!(tensor_layout(20), Some((32, 18))); // IQ4_NL
        let (b16, bytes16) = tensor_layout(16).unwrap();
        // The real expert up-projection is [1024, 2688, 512]; the leading (block)
        // dimension 1024 is a whole number of IQ2_XXS 256-element superblocks.
        assert_eq!(1024 % b16, 0);
        assert_eq!((1024 / b16) * bytes16, 264); // one 1024-wide IQ2_XXS row
        let (b20, bytes20) = tensor_layout(20).unwrap();
        assert_eq!(4096 % b20, 0);
        assert_eq!((4096 / b20) * bytes20, 2304); // one 4096-wide IQ4_NL row
    }

    // ---- split (multi-shard) GGUF fixtures -------------------------------------

    fn split_kv(bytes: &mut Vec<u8>, no: u16, count: u16, tensors: i32) {
        str_field(bytes, "split.no");
        bytes.extend_from_slice(&2u32.to_le_bytes()); // GGUF_TYPE_UINT16
        bytes.extend_from_slice(&no.to_le_bytes());
        str_field(bytes, "split.count");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        str_field(bytes, "split.tensors.count");
        bytes.extend_from_slice(&5u32.to_le_bytes()); // GGUF_TYPE_INT32
        bytes.extend_from_slice(&tensors.to_le_bytes());
    }

    /// A zero-tensor metadata header shard carrying only the split keys.
    fn header_shard(no: u16, count: u16, tensors: i32) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        b.extend_from_slice(&3u64.to_le_bytes()); // kv_count = 3 (split.*)
        split_kv(&mut b, no, count, tensors);
        b
    }

    /// A data shard with F32 tensors and the split keys, laid out exactly as the
    /// single-file indexer expects (aligned cumulative offsets + payload).
    fn data_shard(no: u16, count: u16, tensors_total: i32, tensors: &[(&str, &[u64])]) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes()); // general.alignment + 3 split keys
        str_field(&mut b, "general.alignment");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&32u32.to_le_bytes());
        split_kv(&mut b, no, count, tensors_total);
        let mut rel = 0u64;
        let mut offsets = Vec::new();
        for (_, dims) in tensors {
            offsets.push(rel);
            let elems: u64 = dims.iter().product();
            rel += (elems * 4).div_ceil(32) * 32; // F32 payload, 32-aligned
        }
        let total_data = rel;
        for ((name, dims), off) in tensors.iter().zip(&offsets) {
            str_field(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&0u32.to_le_bytes()); // F32
            b.extend_from_slice(&off.to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&vec![0u8; total_data as usize]);
        b
    }

    #[test]
    fn split_index_accepts_consistent_shards_regardless_of_argument_order() {
        let h = write(&header_shard(0, 3, 3));
        let d1 = write(&data_shard(
            1,
            3,
            3,
            &[
                ("token_embd.weight", &[32, 4]),
                ("output_norm.weight", &[32]),
            ],
        ));
        let d2 = write(&data_shard(2, 3, 3, &[("blk.0.attn_norm.weight", &[32])]));
        // Deliberately out of order: ordering must come from split.no, not args.
        let idx = SplitGgufIndex::inspect(&[d2.clone(), h.clone(), d1.clone()]).unwrap();
        assert_eq!(idx.split_count, 3);
        assert_eq!(idx.total_tensors, 3);
        assert_eq!(idx.indexed_tensor_count(), 3);
        assert_eq!(
            idx.members.iter().map(|m| m.split_no).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(idx.members[0].index.is_none()); // header shard
        let (member, region) = idx.tensor("blk.0.attn_norm.weight").unwrap();
        assert_eq!(member.split_no, 2);
        assert_eq!(region.dimensions, vec![32]);
        assert!(idx.tensor("does.not.exist").is_none());
        for p in [h, d1, d2] {
            std::fs::remove_file(p).unwrap();
        }
    }

    #[test]
    fn split_index_rejects_missing_split_metadata() {
        // A plain single-file GGUF has no split.* keys.
        let plain = write(&fixture(8, 0, 96));
        let err = SplitGgufIndex::inspect(&[plain.clone()]).unwrap_err();
        assert!(err.to_string().contains("split.no"));
        std::fs::remove_file(plain).unwrap();
    }

    #[test]
    fn split_index_rejects_inconsistent_and_incomplete_sets() {
        // Inconsistent split.count between shards.
        let h = write(&header_shard(0, 3, 3));
        let bad = write(&data_shard(1, 2, 3, &[("a.weight", &[32])]));
        assert!(SplitGgufIndex::inspect(&[h.clone(), bad.clone()])
            .unwrap_err()
            .to_string()
            .contains("inconsistent split.count"));
        std::fs::remove_file(&bad).unwrap();

        // split.count disagrees with the number of shards provided.
        assert!(SplitGgufIndex::inspect(&[h.clone()])
            .unwrap_err()
            .to_string()
            .contains("does not match the number of shards"));
        std::fs::remove_file(h).unwrap();

        // Duplicate split.no (gap in the 0..count sequence).
        let a = write(&data_shard(0, 2, 2, &[("a.weight", &[32])]));
        let b = write(&data_shard(0, 2, 2, &[("b.weight", &[32])]));
        assert!(SplitGgufIndex::inspect(&[a.clone(), b.clone()])
            .unwrap_err()
            .to_string()
            .contains("gap or duplicate"));
        for p in [a, b] {
            std::fs::remove_file(p).unwrap();
        }
    }

    #[test]
    fn split_index_rejects_duplicate_tensor_and_count_mismatch() {
        // Same tensor name present in two shards.
        let a = write(&data_shard(0, 2, 2, &[("shared.weight", &[32])]));
        let b = write(&data_shard(1, 2, 2, &[("shared.weight", &[32])]));
        assert!(SplitGgufIndex::inspect(&[a.clone(), b.clone()])
            .unwrap_err()
            .to_string()
            .contains("duplicate tensor name"));
        for p in [a, b] {
            std::fs::remove_file(p).unwrap();
        }

        // Aggregate indexed tensors disagree with split.tensors.count.
        let h = write(&header_shard(0, 2, 5));
        let d = write(&data_shard(
            1,
            2,
            5,
            &[("x.weight", &[32]), ("y.weight", &[32])],
        ));
        assert!(SplitGgufIndex::inspect(&[h.clone(), d.clone()])
            .unwrap_err()
            .to_string()
            .contains("disagrees with split.tensors.count"));
        for p in [h, d] {
            std::fs::remove_file(p).unwrap();
        }
    }
}
