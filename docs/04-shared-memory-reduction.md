# Lesson 4: Shared Memory and Reduction

A reduction turns an array into one value, such as a sum. Threads in a block load values into shared memory, combine pairs, and reduce the number of active values in every round.

## Why synchronize?

```cpp
values[tid] = value;
__syncthreads();
```

Without the barrier, one thread could read a location before another thread writes it. `__syncthreads()` synchronizes threads in one block only, not the complete grid.

## Why produce partial sums?

Different blocks cannot synchronize with `__syncthreads()`. Each block writes one partial sum. The introductory example completes the final addition on the CPU to keep the first reduction easy to follow.

## An invariant of this example

The reduction loop halves `offset` in every round, so the thread count per block must be a power of two. A `static_assert` makes an invalid change, such as 96 or 192 threads, fail during compilation instead of silently returning a partial sum.

## A common trap

Do not let only some threads in a block reach `__syncthreads()` while others skip it. That can cause invalid behavior or a deadlock.
