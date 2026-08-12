# Lesson 1: Grid, Block, Thread, and Warp

## The idea

A kernel is a function executed on the GPU by many threads. Every thread runs the same code but receives different identifiers.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

All indices start at 0:

- `blockIdx.x = 2` identifies the third block.
- `threadIdx.x = 10` identifies the 11th thread inside that block.
- If `blockDim.x = 128`, the global index is `266`, which is the 267th element in human counting.

## The hierarchy

- Grid: the complete kernel launch.
- Block: a group of threads that can share memory and synchronize.
- Thread: one logical instance of the kernel.
- Warp: an execution group of 32 threads on an NVIDIA GPU.

## Why start with `00_device_query`?

Before assuming an A100 is present, ask the CUDA runtime what hardware actually exists. Look for compute capability `8.0` and warp size `32` in the output.

## Check your understanding

With 4 blocks of 256 threads:

- How many logical threads are launched?
- What global index does thread 7 in block 3 receive?
- What is the human-counted position of that element?
