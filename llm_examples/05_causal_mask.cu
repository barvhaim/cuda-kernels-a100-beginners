#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

// Each thread owns one (query_position, key_position) cell.
__global__ void causal_mask(float* mask, int sequence_length) {
  const int key = blockIdx.x * blockDim.x + threadIdx.x;
  const int query = blockIdx.y * blockDim.y + threadIdx.y;
  if (query < sequence_length && key < sequence_length) {
    // Add this FP32 mask to attention scores before softmax.
    mask[query * sequence_length + key] = key <= query ? 0.0f : -INFINITY;
  }
}

int main() {
  constexpr int sequence_length = 4;
  std::vector<float> mask(sequence_length * sequence_length);
  float* d_mask = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * sizeof(float)));
  const dim3 threads(2, 2);
  const dim3 blocks((sequence_length + threads.x - 1) / threads.x,
                    (sequence_length + threads.y - 1) / threads.y);
  causal_mask<<<blocks, threads>>>(d_mask, sequence_length);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(mask.data(), d_mask, mask.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_mask));

  for (int query = 0; query < sequence_length; ++query) {
    for (int key = 0; key < sequence_length; ++key) {
      const float value = mask[query * sequence_length + key];
      const bool visible = value == 0.0f;
      const bool hidden = std::isinf(value) && value < 0.0f;
      if (key <= query ? !visible : !hidden) return 1;
    }
  }
  std::cout << "PASS causal_mask: future token positions are hidden\n";
  return 0;
}
