#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

__global__ void block_sum(const float* input, float* partial, int n) {
  extern __shared__ float values[];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

  float value = 0.0f;
  if (i < n) value += input[i];
  if (i + blockDim.x < n) value += input[i + blockDim.x];
  values[tid] = value;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) values[tid] += values[tid + offset];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = values[0];
}

int main() {
  constexpr int n = 1 << 20;
  // This tree-reduction loop requires a power-of-two block size.
  constexpr int threads = 256;
  static_assert(threads > 0 && (threads & (threads - 1)) == 0,
                "reduction threads must be a power of two");
  const int blocks = (n + threads * 2 - 1) / (threads * 2);
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> input(n, 1.0f), partial(blocks);
  float *d_input = nullptr, *d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

  block_sum<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_partial, n);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  const float result = std::accumulate(partial.begin(), partial.end(), 0.0f);
  const bool correct = std::fabs(result - static_cast<float>(n)) <= 0.5f;
  if (!correct) {
    std::cerr << "Expected " << n << ", got " << result << '\n';
  }
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_partial));
  if (!correct) return 1;
  std::cout << "PASS reduction: sum = " << result << '\n';
  return 0;
}
