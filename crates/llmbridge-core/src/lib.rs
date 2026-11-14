pub mod allocation_admission;
pub mod gguf_index;
pub mod hardware;
#[cfg(feature = "native")]
pub mod inference;
pub mod models;

pub mod f32_chain;
pub mod f32_residual;
pub mod f32_stream;
pub mod f32_swiglu;
pub mod memory_budget;
pub mod io_schedule;
pub mod memory_plan;
#[cfg(feature = "native")]
pub mod q8_batch;
#[cfg(feature = "native")]
pub mod q8_chain;
pub mod runtime_memory;
pub mod storage;
pub mod tensor_batch;
pub mod tensor_cache;
pub mod tensor_layout;

#[cfg(feature = "native")]
pub mod streamed;

pub mod read_cache;

pub mod streamed_plan;

#[cfg(feature = "native")]
pub mod session_performance;

pub mod native_cpu;

#[cfg(target_os = "linux")]
mod cgroup_memory;
