//! Native-derived incremental memory admission. No OS RSS enforcement is claimed.
use crate::hardware::Hardware;
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamedShape {
    pub workspace_fixed_bytes: u64,
    pub workspace_per_token_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub file_bytes: u64,
    pub tensor_payload_bytes: u64,
    pub metadata_prefix_bytes: u64,
    pub training_context: u64,
    pub embedding: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub layers: u64,
    pub feed_forward: u64,
    pub vocabulary: u64,
    pub chunk_bytes: u64,
}
impl StreamedShape {
    pub fn workspace_for_context(&self, context: u64) -> Result<u64, String> {
        self.workspace_per_token_bytes
            .checked_mul(context)
            .and_then(|n| n.checked_add(self.workspace_fixed_bytes))
            .ok_or("native workspace overflow".into())
    }
}
#[derive(Debug, Clone, Copy)]
pub struct StreamedPlanRequest {
    pub context_tokens: u32,
    pub workspace_budget_bytes: u64,
    pub max_additional_bytes: Option<u64>,
    pub cache_budget_bytes: u64,
    pub verification_budget_bytes: u64,
    pub os_reserve_bytes: u64,
    pub runtime_reserve_bytes: u64,
    pub allow_context_reduction: bool,
}
impl Default for StreamedPlanRequest {
    fn default() -> Self {
        Self {
            context_tokens: 128,
            workspace_budget_bytes: 8 * 1024 * 1024,
            max_additional_bytes: None,
            cache_budget_bytes: 0,
            verification_budget_bytes: 0,
            os_reserve_bytes: 512 * 1024 * 1024,
            runtime_reserve_bytes: 32 * 1024 * 1024,
            allow_context_reduction: false,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct StreamedPlan {
    pub requested_context: u32,
    pub selected_context: u32,
    pub native_workspace_bytes: u64,
    pub kv_bytes: u64,
    pub estimated_additional_bytes: u64,
    pub available_after_os_reserve_bytes: u64,
    pub effective_additional_budget_bytes: u64,
    pub metadata_reserve_bytes: u64,
    pub logical_cpus: usize,
}
/// `max_additional_bytes` caps an incremental admission estimate, not total RAM.
/// Exact native workspace coefficients are combined with explicitly estimated reserves.
pub fn plan_streamed_shape(
    shape: StreamedShape,
    hardware: &Hardware,
    request: StreamedPlanRequest,
) -> Result<StreamedPlan, String> {
    if !(64..=8192).contains(&request.context_tokens)
        || request.workspace_budget_bytes == 0
        || request.workspace_budget_bytes > 1024 * 1024 * 1024
        || request.cache_budget_bytes > 32 * 1024 * 1024
        || request.verification_budget_bytes > 64 * 1024 * 1024
        || request.max_additional_bytes == Some(0)
        || shape.kv_bytes_per_token == 0
        || shape.kv_bytes_per_token.checked_add(4) != Some(shape.workspace_per_token_bytes)
        || shape.training_context < 64
        || shape.chunk_bytes == 0
        || shape.metadata_prefix_bytes > shape.file_bytes
        || shape.file_bytes == 0
    {
        return Err("invalid streamed shape or planning request".into());
    }
    let available = hardware
        .available_ram_bytes
        .ok_or("available RAM unknown; refusing streamed admission")?;
    let headroom = available.saturating_sub(request.os_reserve_bytes);
    let budget = request
        .max_additional_bytes
        .map_or(headroom, |n| n.min(headroom));
    let metadata_reserve = shape
        .metadata_prefix_bytes
        .checked_mul(2)
        .ok_or("metadata reserve overflow")?;
    let extra = metadata_reserve
        .checked_add(shape.chunk_bytes)
        .and_then(|n| n.checked_add(request.cache_budget_bytes))
        .and_then(|n| n.checked_add(request.verification_budget_bytes))
        .and_then(|n| n.checked_add(request.runtime_reserve_bytes))
        .ok_or("streamed reserve overflow")?;
    let workspace = budget
        .checked_sub(extra)
        .ok_or("streamed reserves exceed available budget")?
        .min(request.workspace_budget_bytes);
    let fit = workspace
        .checked_sub(shape.workspace_fixed_bytes)
        .ok_or("fixed native workspace does not fit")?
        / shape.workspace_per_token_bytes;
    let selected = u64::from(request.context_tokens)
        .min(shape.training_context)
        .min(fit);
    if selected < 64 {
        return Err("minimum 64-token streamed context does not fit budget".into());
    }
    if selected != u64::from(request.context_tokens) && !request.allow_context_reduction {
        return Err(format!("requested context does not fit model/budget; at most {selected} tokens fit (explicit auto-context required)"));
    }
    let native_workspace_bytes = shape.workspace_for_context(selected)?;
    Ok(StreamedPlan {
        requested_context: request.context_tokens,
        selected_context: selected as u32,
        native_workspace_bytes,
        kv_bytes: shape
            .kv_bytes_per_token
            .checked_mul(selected)
            .ok_or("KV overflow")?,
        estimated_additional_bytes: native_workspace_bytes
            .checked_add(extra)
            .ok_or("total admission overflow")?,
        available_after_os_reserve_bytes: headroom,
        effective_additional_budget_bytes: budget,
        metadata_reserve_bytes: metadata_reserve,
        logical_cpus: hardware.logical_cpus,
    })
}
impl StreamedPlan {
    pub fn to_json(&self, shape: &StreamedShape, request: &StreamedPlanRequest) -> String {
        format!(concat!("{{\"backend\":\"streamed_cpu\",\"native_inspection_metadata_only\":true,\"model_integrity\":\"not_verified_by_plan\",",
            "\"requested_context\":{},\"selected_context\":{},\"context_reduced\":{},\"training_context\":{},",
            "\"native_workspace_bytes\":{},\"workspace_fixed_bytes\":{},\"workspace_per_token_bytes\":{},\"workspace_budget_bytes\":{},",
            "\"kv_bytes\":{},\"kv_bytes_per_token\":{},\"cache_budget_bytes\":{},\"verification_budget_bytes\":{},",
            "\"runtime_reserve_bytes\":{},\"metadata_reserve_bytes\":{},\"reader_buffer_bytes\":{},\"estimated_additional_bytes\":{},",
            "\"effective_additional_budget_bytes\":{},\"available_after_os_reserve_bytes\":{},\"file_bytes\":{},\"tensor_payload_bytes\":{},",
            "\"layers\":{},\"embedding\":{},\"query_heads\":{},\"kv_heads\":{},\"feed_forward\":{},\"vocabulary\":{},\"logical_cpus\":{},\"selected_threads\":1,\"process_limit_enforced_by_plan\":false}}"),
            self.requested_context, self.selected_context, self.selected_context != self.requested_context, shape.training_context,
            self.native_workspace_bytes, shape.workspace_fixed_bytes, shape.workspace_per_token_bytes, request.workspace_budget_bytes,
            self.kv_bytes, shape.kv_bytes_per_token, request.cache_budget_bytes, request.verification_budget_bytes,
            request.runtime_reserve_bytes, self.metadata_reserve_bytes, shape.chunk_bytes, self.estimated_additional_bytes,
            self.effective_additional_budget_bytes, self.available_after_os_reserve_bytes, shape.file_bytes, shape.tensor_payload_bytes, shape.layers, shape.embedding, shape.heads, shape.kv_heads, shape.feed_forward, shape.vocabulary, self.logical_cpus)
    }
}
#[cfg(feature = "native")]
mod native {
    use super::*;
    use crate::{gguf_index::GgufTensorIndex, storage::BoundedLocalReader};
    use std::{
        ffi::{c_char, c_void, CStr},
        path::Path,
        sync::atomic::{AtomicBool, Ordering},
    };
    extern "C" {
        fn llmbridge_abi_version() -> u32;
        fn llmbridge_inspect_streamed_plan_v1(
            size: u64,
            chunk: usize,
            read: extern "C" fn(*mut c_void, *mut c_void, u64, usize) -> usize,
            poll: extern "C" fn(*mut c_void) -> i32,
            state: *mut c_void,
            out: *mut StreamedShape,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
    }
    struct State<'a> {
        reader: BoundedLocalReader,
        cancelled: &'a AtomicBool,
        limit: u64,
        failure: Option<String>,
    }
    extern "C" fn poll(state: *mut c_void) -> i32 {
        if state.is_null() {
            return 0;
        }
        // SAFETY: the synchronous native call borrows a live State only.
        let s = unsafe { &*(state.cast::<State<'_>>()) };
        i32::from(!s.cancelled.load(Ordering::Acquire))
    }
    extern "C" fn read(state: *mut c_void, out: *mut c_void, offset: u64, count: usize) -> usize {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if state.is_null() || out.is_null() || count == 0 {
                return 0;
            }
            let s = unsafe { &mut *(state.cast::<State<'_>>()) };
            match offset.checked_add(count as u64) {
                Some(end) if end <= s.limit => {}
                _ => {
                    s.failure = Some("plan reader attempted to leave GGUF metadata".into());
                    return 0;
                }
            }
            match s
                .reader
                .read_range_cancellable(offset, count, || s.cancelled.load(Ordering::Acquire))
            {
                Ok(bytes) => {
                    if s.cancelled.load(Ordering::Acquire) {
                        return 0;
                    }
                    // SAFETY: native supplies count writable non-overlapping bytes.
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast(), count);
                    }
                    count
                }
                Err(error) => {
                    s.failure = Some(error.to_string());
                    0
                }
            }
        }))
        .unwrap_or(0)
    }
    pub fn inspect_streamed_shape(
        model: &Path,
        chunk: usize,
        cancelled: &AtomicBool,
    ) -> Result<StreamedShape, String> {
        inspect(model, chunk, cancelled, None)
    }
    pub fn inspect_streamed_shape_verified(
        model: &Path,
        chunk: usize,
        cancelled: &AtomicBool,
        expected: &str,
        budget: usize,
    ) -> Result<StreamedShape, String> {
        inspect(model, chunk, cancelled, Some((expected, budget)))
    }
    fn inspect(
        model: &Path,
        chunk: usize,
        cancelled: &AtomicBool,
        integrity: Option<(&str, usize)>,
    ) -> Result<StreamedShape, String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("streamed plan cancelled before inspection".into());
        }
        if !(34..=1024 * 1024).contains(&chunk) {
            return Err("invalid plan chunk".into());
        }
        let index = GgufTensorIndex::inspect(model).map_err(|e| e.to_string())?;
        let mut reader =
            BoundedLocalReader::open(model, chunk, chunk * 2).map_err(|e| e.to_string())?;
        if reader.identity() != &index.source_identity {
            return Err("model changed before plan inspection".into());
        }
        // Verified auto-planning authenticates before exposing metadata to native parsing.
        let _sealed = integrity
            .map(|(expected, budget)| {
                reader.authenticate_model(expected, index.data_offset, budget, || {
                    cancelled.load(Ordering::Acquire)
                })
            })
            .transpose()
            .map_err(|e| e.to_string())?;
        if unsafe { llmbridge_abi_version() } != 1 {
            return Err("native plan ABI mismatch".into());
        }
        let mut s = State {
            reader,
            cancelled,
            limit: index.data_offset,
            failure: None,
        };
        let mut out = StreamedShape::default();
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            llmbridge_inspect_streamed_plan_v1(
                index.file_bytes,
                chunk,
                read,
                poll,
                (&mut s as *mut State<'_>).cast(),
                &mut out,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if cancelled.load(Ordering::Acquire) || status == 2 {
            return Err("streamed plan cancelled".into());
        }
        if status != 0 {
            return Err(format!(
                "native plan failed: {}; source: {}",
                unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy(),
                s.failure.as_deref().unwrap_or("no read error")
            ));
        }
        s.reader.validate_source().map_err(|e| e.to_string())?;
        if out.file_bytes != index.file_bytes || out.metadata_prefix_bytes != index.data_offset {
            return Err("native/Rust plan disagreement".into());
        }
        Ok(out)
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn probe_callback_rejects_weight_ranges_and_cancelled_reads() {
            let path = std::env::temp_dir()
                .join(format!("llmbridge-plan-callback-{}", std::process::id()));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            std::io::Write::write_all(&mut file, b"abcdefghijkl").unwrap();
            let flag = AtomicBool::new(false);
            let reader = BoundedLocalReader::open(&path, 8, 16).unwrap();
            let mut state = State {
                reader,
                cancelled: &flag,
                limit: 4,
                failure: None,
            };
            let mut destination = [0xaau8; 8];
            let pointer = (&mut state as *mut State<'_>).cast();
            assert_eq!(read(pointer, destination.as_mut_ptr().cast(), 2, 4), 0);
            assert_eq!(state.reader.disk_bytes_read(), 0);
            assert_eq!(destination, [0xaa; 8]);
            assert_eq!(read(pointer, destination.as_mut_ptr().cast(), 1, 2), 2);
            assert_eq!(&destination[..2], b"bc");
            flag.store(true, Ordering::Release);
            destination.fill(0x55);
            assert_eq!(read(pointer, destination.as_mut_ptr().cast(), 0, 2), 0);
            assert_eq!(state.reader.disk_bytes_read(), 2);
            assert_eq!(destination, [0x55; 8]);
            drop(state);
            drop(file);
            std::fs::remove_file(path).unwrap();
        }
        #[test]
        fn precancelled_probe_never_opens_the_model() {
            assert!(inspect_streamed_shape(
                Path::new("/absent.gguf"),
                65536,
                &AtomicBool::new(true)
            )
            .unwrap_err()
            .contains("cancelled before inspection"));
        }
    }
}
#[cfg(feature = "native")]
pub use native::{inspect_streamed_shape, inspect_streamed_shape_verified};
#[cfg(test)]
mod tests {
    use super::*;
    fn shape() -> StreamedShape {
        StreamedShape {
            workspace_fixed_bytes: 100,
            workspace_per_token_bytes: 10,
            kv_bytes_per_token: 6,
            training_context: 512,
            file_bytes: 10000,
            metadata_prefix_bytes: 16,
            chunk_bytes: 8,
            ..Default::default()
        }
    }
    fn hw(available: Option<u64>) -> Hardware {
        Hardware {
            logical_cpus: 2,
            total_ram_bytes: Some(2 * 1024 * 1024 * 1024),
            available_ram_bytes: available,
        }
    }
    fn req() -> StreamedPlanRequest {
        StreamedPlanRequest {
            workspace_budget_bytes: 2000,
            runtime_reserve_bytes: 0,
            os_reserve_bytes: 0,
            ..Default::default()
        }
    }
    #[test]
    fn exact_largest_context_boundary() {
        let r = StreamedPlanRequest {
            workspace_budget_bytes: 1060,
            allow_context_reduction: true,
            ..req()
        };
        let p = plan_streamed_shape(shape(), &hw(Some(5000)), r).unwrap();
        assert_eq!(p.selected_context, 96);
        assert_eq!(p.native_workspace_bytes, 1060);
        assert_eq!(p.kv_bytes, 576);
        assert!(shape().workspace_for_context(97).unwrap() > r.workspace_budget_bytes);
    }
    #[test]
    fn opt_in_and_minimum_are_required() {
        assert!(plan_streamed_shape(
            shape(),
            &hw(Some(5000)),
            StreamedPlanRequest {
                workspace_budget_bytes: 1060,
                ..req()
            }
        )
        .is_err());
        assert!(plan_streamed_shape(
            shape(),
            &hw(Some(5000)),
            StreamedPlanRequest {
                workspace_budget_bytes: 739,
                allow_context_reduction: true,
                ..req()
            }
        )
        .is_err());
    }
    #[test]
    fn unknown_memory_and_exhausted_reserves_fail_closed() {
        assert!(plan_streamed_shape(shape(), &hw(None), req()).is_err());
        assert!(plan_streamed_shape(
            shape(),
            &hw(Some(5000)),
            StreamedPlanRequest {
                os_reserve_bytes: 5001,
                ..req()
            }
        )
        .is_err());
        assert!(plan_streamed_shape(
            shape(),
            &hw(Some(5000)),
            StreamedPlanRequest {
                max_additional_bytes: Some(779),
                ..req()
            }
        )
        .is_err());
    }
    #[test]
    fn all_reserves_share_the_incremental_budget() {
        let r = StreamedPlanRequest {
            max_additional_bytes: Some(2020),
            cache_budget_bytes: 100,
            verification_budget_bytes: 200,
            runtime_reserve_bytes: 300,
            ..req()
        };
        assert_eq!(
            plan_streamed_shape(shape(), &hw(Some(5000)), r)
                .unwrap()
                .estimated_additional_bytes,
            2020
        );
        assert!(plan_streamed_shape(
            shape(),
            &hw(Some(5000)),
            StreamedPlanRequest {
                max_additional_bytes: Some(2019),
                ..r
            }
        )
        .is_err());
    }
    #[test]
    fn training_context_is_respected() {
        let mut s = shape();
        s.training_context = 96;
        assert!(plan_streamed_shape(s, &hw(Some(5000)), req()).is_err());
        assert_eq!(
            plan_streamed_shape(
                s,
                &hw(Some(5000)),
                StreamedPlanRequest {
                    allow_context_reduction: true,
                    ..req()
                }
            )
            .unwrap()
            .selected_context,
            96
        );
    }
    #[test]
    fn overflow_and_invalid_coefficients_do_not_admit() {
        let mut s = shape();
        s.workspace_per_token_bytes = u64::MAX;
        assert!(s.workspace_for_context(2).is_err());
        assert!(plan_streamed_shape(s, &hw(Some(u64::MAX)), req()).is_err());
        s = shape();
        s.workspace_fixed_bytes = u64::MAX;
        assert!(plan_streamed_shape(s, &hw(Some(u64::MAX)), req()).is_err());
    }
}
