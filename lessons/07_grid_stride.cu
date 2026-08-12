#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void saxpy(float alpha, const float* x, float* y, int n) {
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = start; i < n; i += stride) {
    y[i] = alpha * x[i] + y[i];
  }
}

int main() {
  constexpr int n = 1 << 24;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> x(n, 2.0f), y(n, 1.0f);
  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  constexpr int threads = 256;
  const int blocks = 32 * properties.multiProcessorCount;
  saxpy<<<blocks, threads>>>(3.0f, d_x, d_y, n);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  bool correct = true;
  for (int i = 0; i < n; ++i) {
    if (std::fabs(y[i] - 7.0f) > 1e-6f) {
      std::cerr << "Mismatch at " << i << '\n';
      correct = false;
      break;
    }
  }
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  if (!correct) return 1;
  std::cout << "PASS grid-stride SAXPY using " << blocks << " blocks\n";
  return 0;
}
