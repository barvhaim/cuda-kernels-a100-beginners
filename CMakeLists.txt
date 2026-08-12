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

add_cuda_lesson(00_device_query lessons/00_device_query.cu)
add_cuda_lesson(01_vector_add lessons/01_vector_add.cu)
add_cuda_lesson(02_grid_stride lessons/02_grid_stride.cu)
add_cuda_lesson(03_reduction lessons/03_reduction.cu)
add_cuda_lesson(04_tiled_matmul lessons/04_tiled_matmul.cu)
