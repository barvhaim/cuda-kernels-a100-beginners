# Exercises

Work in order. For every exercise: predict first, change one thing, run it, and record what happened.

## Part A: Foundations

### Exercise 1: Start at zero

In `02_one_block_index`, launch 8 threads. Before running, write down the first index, the last index, and the human-counted position of `threadIdx.x=7`.

### Exercise 2: Global index

In `03_global_index`, launch 3 blocks with 5 threads in each block. Manually compute the index of thread 2 in block 1 and the total number of threads.

### Exercise 3: Extra threads

In `04_bounds_check`, set `n=10` and `threads=4`. Predict the block count, launched-thread count, and extra-thread count.

### Exercise 4: Host and device

In `05_memory_roundtrip`, change the kernel from `+1` to `*2`. Update the expected output only after you have observed the test fail.

### Exercise 5: Vector addition

Change `06_vector_add` to use 1,000 elements and 128 threads per block. Print the block count and explain why more than 1,000 threads are launched.

## Part B: CUDA patterns

### Exercise 6: Remove the guard on purpose

Create a local copy of vector addition, remove `if (i < n)`, and run it only under `compute-sanitizer`. Do not keep the broken version. Record what the tool reports.

### Exercise 7: Grid-stride SAXPY

Change `alpha` and the input values in `07_grid_stride`. Update the expected result and verify that the test fails before the update and passes afterward.

### Exercise 8: Reduction

Set the input values in `08_reduction` to `2.0f`. The kernel requires a power-of-two `threads` value. Do not change it to 96 or 192. Then replace the CPU completion with repeated launches using `threads=256` until only one sum remains.

### Exercise 9: Matrix multiplication

Try `TILE=8`, `16`, and `32` in `09_tiled_matmul`. Verify correctness and measure each version. Do not assume the largest tile will be fastest.

## Part C: LLM kernels

### Exercise 10: Embedding lookup

Add a fifth token to the table in `llm_01_token_embedding` and change the token IDs. Draw the shape of the table and output. Before trying an out-of-range token ID, add CPU validation that rejects the launch. Verify that invalid input is rejected safely.

### Exercise 11: Residual connection

Change the hidden vector length to 13 and the block size to 8. Compute the number of extra threads and explain why the guard is required.

### Exercise 12: SiLU and SwiGLU

Run SiLU on 9 values with 4 threads. Then add a second vector called `gate` and compute an educational version of:

```text
output[i] = SiLU(up[i]) * gate[i]
```

### Exercise 13: RMSNorm

Change the weights so they are not all 1. Compare with a Python reference. Explain which values are shared by all threads and which values are private to one thread.

### Exercise 14: Causal mask

Change the sequence length to 5. Verify that a grid of 2x2-thread blocks still covers all 25 cells through two-dimensional bounds checks.

### Exercise 15: Stable softmax

Temporarily remove maximum subtraction while keeping scores near 1000. Record the resulting `inf` or `nan`, restore the fix, and run again.

### Exercise 16: Q projection

Change `llm_07_linear_projection` from output size 2 to output size 3. Write a Python reference and compare every output dimension.

### Exercise 17: Mini Transformer step

Print or copy back to the host the hidden state after embedding, after residual addition, and after RMSNorm. Record the shape and values at every boundary.

### Exercise 18: What should be fused?

Choose two adjacent operations in the mini pipeline. Explain which global-memory writes and reads a fused kernel could avoid, and describe the cost in complexity and testing.

## A100 experiment

Run Nsight Compute on `06_vector_add`, `llm_04_rmsnorm`, and `llm_07_linear_projection`. For each kernel, predict whether it is memory-bound or compute-bound and cite two metrics that support your conclusion.

## Progress log

For every exercise, record:

- The command you ran.
- The shape of every input and output.
- The expected and actual result.
- One failure you observed and its cause.
- One concept you can now explain without looking at the code.
