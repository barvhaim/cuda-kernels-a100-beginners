#include "cuda_check.cuh"

#include <iostream>
#include <vector>

// Each thread owns one (query_position, key_position) cell.
__global__ void causal_mask(float* mask, int sequence_length) {
  const int key = blockIdx.x * blockDim.x + threadIdx.x;
  const int query = blockIdx.y * blockDim.y + threadIdx.y;
  if (query < sequence_length && key < sequence_length) {
    mask[query * sequence_length + key] = key <= query ? 0.0f : -1.0e9f;
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
      const bool visible = mask[query * sequence_length + key] == 0.0f;
      if (visible != (key <= query)) return 1;
    }
  }
  std::cout << "PASS causal_mask: future token positions are hidden\n";
  return 0;
}
