//! Integration regressions for indexed tensor boundaries and cooperative cancellation.
//! No native inference feature or model download is required.
use llmbridge_core::gguf_index::GgufTensorIndex;
use llmbridge_core::storage::BoundedLocalReader;
use std::fs::{self, File};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "llmbridge-tensor-boundaries-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata count
        for (name, offset) in [("first", 0u64), ("second", 32u64)] {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes()); // rank
            bytes.extend_from_slice(&4u64.to_le_bytes()); // four F32 values
            bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0x11; 16]);
        bytes.extend_from_slice(&[0xEE; 16]); // padding must never be streamed
        bytes.extend_from_slice(&[0x22; 16]);
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn indexed_streams_never_include_neighbor_or_alignment_padding() {
    let fixture = Fixture::new();
    let index = GgufTensorIndex::inspect(&fixture.0).unwrap();
    let mut reader = BoundedLocalReader::open(&fixture.0, 7, 14).unwrap();
    for (name, expected) in [("first", 0x11u8), ("second", 0x22u8)] {
        let mut received = Vec::new();
        let length = index
            .stream_tensor(&mut reader, name, |chunk| {
                assert!(chunk.len() <= 7);
                received.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(length, 16);
        assert_eq!(received, vec![expected; 16]);
    }
    assert_eq!(reader.disk_bytes_read(), 32);
    let before = reader.disk_bytes_read();
    assert_eq!(
        index
            .stream_tensor(&mut reader, "missing", |_| Ok(()))
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(reader.disk_bytes_read(), before);
}

#[test]
fn cancellation_between_chunks_does_not_read_another_chunk() {
    let fixture = Fixture::new();
    let index = GgufTensorIndex::inspect(&fixture.0).unwrap();
    let mut reader = BoundedLocalReader::open(&fixture.0, 7, 14).unwrap();
    let cancelled = AtomicBool::new(true);
    let mut calls = 0usize;
    let err = index
        .stream_tensor_cancellable(&mut reader, "first", &cancelled, |_| {
            calls += 1;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Interrupted);
    assert_eq!(calls, 0);
    assert_eq!(reader.disk_bytes_read(), 0);

    cancelled.store(false, Ordering::Release);
    let err = index
        .stream_tensor_cancellable(&mut reader, "first", &cancelled, |_| {
            calls += 1;
            cancelled.store(true, Ordering::Release);
            Ok(())
        })
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Interrupted);
    assert_eq!(calls, 1);
    assert_eq!(reader.disk_bytes_read(), 7);
}
