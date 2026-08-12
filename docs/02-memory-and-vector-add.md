# Lesson 2: Memory and Vector Addition

## The path

```text
host vectors -> cudaMemcpy HostToDevice -> kernel -> cudaMemcpy DeviceToHost -> verification
```

`cudaMalloc` allocates device memory. The kernel receives `d_a`, `d_b`, and `d_c`, not the CPU vectors.

## The kernel

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

Round the number of blocks upward:

```cpp
int blocks = (n + threads - 1) / threads;
```

This can launch extra threads. The `i < n` condition prevents out-of-bounds access.

## Asynchronous execution

A kernel launch is usually asynchronous with respect to the CPU. `CudaTimer` uses CUDA events, while a synchronous copy back to the host waits for the required work to complete.

## Reading task

Open `lessons/06_vector_add.cu` and identify six stages: host allocation, device allocation, H2D copy, launch, D2H copy, and verification/free.
