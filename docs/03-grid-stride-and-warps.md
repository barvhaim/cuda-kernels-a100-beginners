# Lesson 3: Grid-Stride Loops and Warps

## Why not always launch one thread per element?

You can launch a relatively small grid and let every thread process multiple elements:

```cpp
int start = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = start; i < n; i += stride) {
  y[i] = alpha * x[i] + y[i];
}
```

The same kernel can now handle different input sizes, while the block count can be chosen according to the number of SMs.

## Memory coalescing

Adjacent threads in a warp access adjacent indices in this example. This pattern is friendly to the memory system. An access pattern such as `x[i * 32]` is usually less efficient.

## Warp divergence

Threads in a warp execute instructions together. If half enter one branch and half enter another, the paths may execute separately. A short boundary check at the edge of a grid is normal; heavy branching based on `threadIdx.x` deserves careful analysis.

## On A100

The number of SMs differs between A100 variants, so the lesson reads it at runtime instead of hard-coding a value.
