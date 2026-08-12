#include "cuda_check.cuh"

#include <cstdio>
#include <iostream>

__global__ void show_bounds(int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    printf("global=%d works on data[%d]\n", i, i);
  } else {
    printf("global=%d is extra and does no work\n", i);
  }
}

int main() {
  constexpr int n = 6;
  constexpr int threads = 4;
  const int blocks = (n + threads - 1) / threads;
  show_bounds<<<blocks, threads>>>(n);
  check_kernel();
  std::cout << "PASS bounds_check: launched 8 threads for 6 values\n";
  return 0;
}
