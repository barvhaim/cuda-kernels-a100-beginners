#include "cuda_check.cuh"

#include <cstdio>
#include <iostream>

__global__ void hello_kernel() {
  printf("Hello from the GPU!\n");
}

int main() {
  std::cout << "Hello from the CPU!\n";
  hello_kernel<<<1, 1>>>();
  check_kernel();
  std::cout << "PASS hello_kernel\n";
  return 0;
}
