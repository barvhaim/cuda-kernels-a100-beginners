# Lesson 5: Tiled Matrix Multiplication

In naive matrix multiplication, values from A and B are loaded from global memory repeatedly. Tiling loads small regions into shared memory and reuses them.

## Two-dimensional mapping

```cpp
int row = blockIdx.y * TILE + threadIdx.y;
int col = blockIdx.x * TILE + threadIdx.x;
```

Each block computes one tile of C. In every phase, it loads one tile from A and one from B, synchronizes, performs multiply-add operations, and synchronizes before loading the next tiles.

## Why is N equal to 257?

The odd size verifies that the kernel does not assume the dimension is divisible by 16. Out-of-range loads become 0, and the output write has a bounds check.

## What this example does not do

It does not use Tensor Cores and is not a competitive matrix-multiplication kernel. Real A100 workloads normally use cuBLAS or CUTLASS. This lesson exists to teach data reuse, synchronization, and two-dimensional indexing.
