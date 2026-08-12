#include "cuda_check.cuh"
#include "cuda_timer.cuh"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int n = 1 << 20;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> a(n, 1.5f), b(n, 2.5f), c(n);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  CudaTimer timer;
  timer.start();
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  const float elapsed_ms = timer.stop_ms();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  bool correct = true;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(c[i] - 4.0f) > 1e-6f) {
      std::cerr << "Mismatch at " << i << '\n';
      correct = false;
      break;
    }
  }
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  if (!correct) return 1;
  std::cout << "PASS vector_add: " << n << " values in " << elapsed_ms << " ms\n";
  return 0;
}
