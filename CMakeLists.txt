cmake_minimum_required(VERSION 3.24)
project(cuda_kernels_a100_beginners LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# NVIDIA A100 uses compute capability 8.0 (sm_80).
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(add_cuda_lesson target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--lineinfo>)
endfunction()

# Foundations: one new concept per executable.
add_cuda_lesson(00_device_query lessons/00_device_query.cu)
add_cuda_lesson(01_hello_kernel lessons/01_hello_kernel.cu)
add_cuda_lesson(02_one_block_index lessons/02_one_block_index.cu)
add_cuda_lesson(03_global_index lessons/03_global_index.cu)
add_cuda_lesson(04_bounds_check lessons/04_bounds_check.cu)
add_cuda_lesson(05_memory_roundtrip lessons/05_memory_roundtrip.cu)
add_cuda_lesson(06_vector_add lessons/06_vector_add.cu)
add_cuda_lesson(07_grid_stride lessons/07_grid_stride.cu)
add_cuda_lesson(08_reduction lessons/08_reduction.cu)
add_cuda_lesson(09_tiled_matmul lessons/09_tiled_matmul.cu)

# Educational LLM connections. These favor clarity over production performance.
add_cuda_lesson(llm_01_token_embedding llm_examples/01_token_embedding.cu)
add_cuda_lesson(llm_02_residual_add llm_examples/02_residual_add.cu)
add_cuda_lesson(llm_03_silu_activation llm_examples/03_silu_activation.cu)
add_cuda_lesson(llm_04_rmsnorm llm_examples/04_rmsnorm.cu)
add_cuda_lesson(llm_05_causal_mask llm_examples/05_causal_mask.cu)
add_cuda_lesson(llm_06_attention_softmax llm_examples/06_attention_softmax.cu)
add_cuda_lesson(llm_07_linear_projection llm_examples/07_linear_projection.cu)
add_cuda_lesson(llm_08_mini_transformer_step llm_examples/08_mini_transformer_step.cu)
