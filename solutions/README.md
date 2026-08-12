# Solution Guidance

Try the exercises yourself first. This file provides the key direction and invariant, not complete code to copy.

## Foundations

### Exercise 1

With 8 threads, the indices are 0 through 7. `threadIdx.x=7` is the eighth thread in human counting.

### Exercise 2

```text
global = 1 * 5 + 2 = 7
```

Three blocks multiplied by five threads equals 15 threads.

### Exercise 3

```text
blocks = ceil(10 / 4) = 3
launched = 3 * 4 = 12
extra = 2
```

### Exercise 4

The expected output for `{10,20,30,40}` after multiplication is `{20,40,60,80}`. Observing the test fail before updating the expectation proves that the check is sensitive to the change.

### Exercise 5

```text
blocks = ceil(1000 / 128) = 8
launched = 1024
```

Indices 1000 through 1023 must stop at the guard.

## CUDA patterns

### Exercise 6

`compute-sanitizer` should report an invalid global-memory access. A normal run that does not crash is not proof of correctness.

### Exercise 7

The formula is `y = alpha * x + y`. Compute a new expected result on the CPU before comparing the output.

### Exercise 8

If there are N values equal to 2, the sum is `2*N`. In every new launch, the input count equals the number of partial sums produced by the previous launch. The thread count remains a power of two.

### Exercise 9

Tile size changes block size, shared-memory use, register use, and occupancy. There is no universal winning size; measure only after verifying correctness.

## LLM kernels

### Exercise 10

The embedding table has shape `[vocab_size, hidden_size]`. A token ID must satisfy `0 <= id < vocab_size` before launch; do not rely on a kernel reading beyond the table.

### Exercise 11

```text
blocks = ceil(13 / 8) = 2
launched = 16
extra = 3
```

### Exercise 12

An educational SwiGLU combines two vectors:

```text
output[i] = SiLU(up[i]) * gate[i]
```

Each thread can compute one element independently.

### Exercise 13

`inverse_rms` is shared across the hidden vector; `x[i]` and `weight[i]` belong to one dimension. The reduction creates the shared statistic.

### Exercise 14

A grid of `ceil(5/2) x ceil(5/2) = 3 x 3` blocks, with 2x2 threads per block, launches 36 threads for 25 cells. Both bounds-check dimensions are required.

### Exercise 15

`exp(1000)` overflows in float. After subtracting `max=1003`, the arguments are `-3,-2,-1,0`, so the exponentials remain finite.

### Exercise 16

Every output dimension is a different dot product between the input and one column of the weight matrix. The Python reference must use the same row-major layout as the CUDA code.

### Exercise 17

The shapes remain `[hidden_size]` through embedding, residual addition, and RMSNorm. Only the LM head changes the result to `[vocab_size]` logits.

### Exercise 18

Fusion can avoid writing an intermediate to global memory and reading it back. The cost is a more complex kernel, increased register pressure, reduced modularity, and harder testing.
