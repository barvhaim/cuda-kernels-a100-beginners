#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

constexpr int TILE = 16;

__global__ void tiled_matmul(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[TILE][TILE];
  __shared__ float tile_b[TILE][TILE];
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float value = 0.0f;

  for (int tile = 0; tile < (n + TILE - 1) / TILE; ++tile) {
    const int a_col = tile * TILE + threadIdx.x;
    const int b_row = tile * TILE + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
      value += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < n && col < n) c[row * n + col] = value;
}

int main() {
  constexpr int n = 257;  // Intentionally not divisible by TILE.
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
  std::vector<float> a(n * n, 1.0f), b(n * n, 1.0f), c(n * n);
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 threads(TILE, TILE);
  const dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
  tiled_matmul<<<blocks, threads>>>(d_a, d_b, d_c, n);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  for (float value : c) {
    if (std::fabs(value - static_cast<float>(n)) > 1e-3f) {
      std::cerr << "Mismatch: expected " << n << ", got " << value << '\n';
      return 1;
    }
  }
  std::cout << "PASS tiled_matmul: " << n << "x" << n << '\n';
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}
