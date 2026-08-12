#pragma once

#include "cuda_check.cuh"

class CudaTimer {
 public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { CUDA_CHECK(cudaEventRecord(start_)); }

  float stop_ms() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};
