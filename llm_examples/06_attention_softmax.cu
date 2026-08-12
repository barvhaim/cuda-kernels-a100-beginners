#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

// Educational stable softmax for one attention row.
// Subtracting the row maximum prevents exp(score) from overflowing.
__global__ void softmax_row(const float* scores, float* probabilities, int n) {
  extern __shared__ float storage[];
  float* maxima = storage;
  float* exponentials = storage + blockDim.x;
  const int tid = threadIdx.x;

  maxima[tid] = tid < n ? scores[tid] : -INFINITY;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) maxima[tid] = fmaxf(maxima[tid], maxima[tid + offset]);
    __syncthreads();
  }

  exponentials[tid] = tid < n ? expf(scores[tid] - maxima[0]) : 0.0f;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) exponentials[tid] += exponentials[tid + offset];
    __syncthreads();
  }
  if (tid < n) probabilities[tid] = expf(scores[tid] - maxima[0]) / exponentials[0];
}

int main() {
  constexpr int n = 4;
  constexpr int threads = 4;
  static_assert((threads & (threads - 1)) == 0, "threads must be a power of two");
  static_assert(threads >= n,
                "educational one-block softmax needs one thread per score");
  const std::vector<float> scores{1000.0f, 1001.0f, 1002.0f, 1003.0f};
  std::vector<float> probabilities(n);
  float *d_scores = nullptr, *d_probabilities = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_probabilities, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_scores, scores.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  softmax_row<<<1, threads, 2 * threads * sizeof(float)>>>(d_scores, d_probabilities, n);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(probabilities.data(), d_probabilities, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_probabilities));

  float sum = 0.0f;
  for (float p : probabilities) {
    if (!std::isfinite(p)) return 1;
    sum += p;
  }
  if (std::fabs(sum - 1.0f) > 1e-5f || probabilities[3] <= probabilities[2]) return 1;
  std::cout << "PASS attention_softmax: stable probabilities sum to 1\n";
  return 0;
}
