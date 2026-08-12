# CUDA Foundations, Step by Step

The goal of this chapter is not speed. The goal is to understand every line before moving forward.

## The minimum mental model

- The CPU is the `host`.
- The GPU is the `device`.
- A function declared with `__global__` is a kernel launched by the CPU and executed on the GPU.
- `kernel<<<blocks, threads>>>(...)` chooses how many logical threads to launch.
- Indexing starts at 0.

## 1. One kernel, one thread

File: `lessons/01_hello_kernel.cu`

```cpp
__global__ void hello_kernel() {
  printf("Hello from the GPU!\n");
}

hello_kernel<<<1, 1>>>();
```

This launch contains one block with one thread. For now, focus only on crossing the CPU-to-GPU boundary.

Run it:

```bash
./build/01_hello_kernel
```

## 2. Indexing inside one block

File: `lessons/02_one_block_index.cu`

```cpp
show_thread_index<<<1, 4>>>();
```

The four threads receive `threadIdx.x` values 0, 1, 2, and 3. The order of the `printf` lines is not guaranteed because the threads execute concurrently.

**LLM connection:** each thread could own one dimension of a small hidden state.

## 3. Global indexing

File: `lessons/03_global_index.cu`

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Two blocks with four threads each produce indices 0 through 7:

```text
Block 0: 0 1 2 3
Block 1: 4 5 6 7
```

**LLM connection:** an activation tensor is usually larger than one block, so every thread needs a unique index across the complete grid.

## 4. Extra threads and bounds checks

File: `lessons/04_bounds_check.cu`

For six values and four threads per block:

```cpp
blocks = (6 + 4 - 1) / 4;  // 2 blocks
```

CUDA launches eight threads. Indices 6 and 7 are extra and must stop at a guard:

```cpp
if (i < n) {
  // safe access to data[i]
}
```

**LLM connection:** hidden size, token count, and vocabulary size do not have to divide evenly by the block size.

## 5. Memory: host to device and back

File: `lessons/05_memory_roundtrip.cu`

The data path is:

```text
CPU vector
  -> cudaMalloc
  -> cudaMemcpy HostToDevice
  -> kernel adds 1
  -> cudaMemcpy DeviceToHost
  -> CPU verifies result
  -> cudaFree
```

**LLM connection:** token IDs, weights, and activations must live in memory accessible to the GPU while a kernel executes.

## 6. Vector addition

File: `lessons/06_vector_add.cu`

Now combine indexing, a bounds check, and memory management into one complete program. Each thread computes one element:

```cpp
c[i] = a[i] + b[i];
```

**LLM connection:** this is the same pattern as a residual connection:

```text
hidden = layer_output + residual
```

## Before you continue

You are ready for grid-stride loops only if you can answer these questions without looking:

1. Why is `threadIdx.x = 10` the 11th thread?
2. Why does `blockIdx.x * blockDim.x` appear in the global-index formula?
3. Why do we sometimes launch more threads than values?
4. What is the difference between `host_values` and `device_values`?
5. Why should learning code check errors and synchronize after a kernel launch?
