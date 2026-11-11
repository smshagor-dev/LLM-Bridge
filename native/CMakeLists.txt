cmake_minimum_required(VERSION 3.22)
project(llmbridge_native LANGUAGES C CXX)
include(FetchContent)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG cd959bae99ff4bed22bef2604f45241c09bbb8c8
  GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(llama_cpp)
add_library(llmbridge_native STATIC src/bridge.cpp)
target_compile_features(llmbridge_native PRIVATE cxx_std_20)
target_include_directories(llmbridge_native PUBLIC include)
target_link_libraries(llmbridge_native PRIVATE llama ggml)
install(TARGETS llmbridge_native ARCHIVE DESTINATION lib)
