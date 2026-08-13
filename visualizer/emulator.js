const EXAMPLES = {
  indexing: {
    name: "Global indexing",
    lesson: "lessons/03_global_index.cu",
    code: [
      "const int i = blockIdx.x * blockDim.x + threadIdx.x;",
      "printf(\"block=%d thread=%d global=%d\\n\", ...);",
    ],
  },
  bounds: {
    name: "Bounds check",
    lesson: "lessons/04_bounds_check.cu",
    code: [
      "const int i = blockIdx.x * blockDim.x + threadIdx.x;",
      "if (i < n) { work(data[i]); }",
    ],
  },
  "residual-add": {
    name: "LLM residual add",
    lesson: "llm_examples/02_residual_add.cu",
    code: [
      "const int i = blockIdx.x * blockDim.x + threadIdx.x;",
      "if (i < n) hidden_out[i] = layer_output[i] + residual[i];",
    ],
  },
  "vector-add": {
    name: "Vector addition",
    lesson: "lessons/06_vector_add.cu",
    code: [
      "const int i = blockIdx.x * blockDim.x + threadIdx.x;",
      "if (i < n) c[i] = a[i] + b[i];",
    ],
  },
  "grid-stride": {
    name: "Grid-stride loop",
    lesson: "lessons/07_grid_stride.cu",
    code: [
      "int i = blockIdx.x * blockDim.x + threadIdx.x;",
      "for (; i < n; i += blockDim.x * gridDim.x)",
      "  output[i] = input[i] * 2;",
    ],
  },
};

const clampInteger = (value, min, max) =>
  Math.min(max, Math.max(min, Math.round(Number(value) || min)));

export function normalizeConfig(config = {}) {
  return {
    example: EXAMPLES[config.example] ? config.example : "residual-add",
    blocks: clampInteger(config.blocks, 1, 8),
    threadsPerBlock: clampInteger(config.threadsPerBlock, 1, 16),
    dataSize: clampInteger(config.dataSize, 1, 48),
  };
}

export function createSimulation(config = {}) {
  const normalized = normalizeConfig(config);
  const { example, blocks, threadsPerBlock, dataSize } = normalized;
  const launchedThreads = blocks * threadsPerBlock;
  const inputA = Array.from({ length: dataSize }, (_, i) => i + 1);
  const inputB = Array.from({ length: dataSize }, (_, i) => (i + 1) * 10);
  const output = Array(dataSize).fill(null);

  const threads = Array.from({ length: launchedThreads }, (_, globalIdx) => {
    const blockIdx = Math.floor(globalIdx / threadsPerBlock);
    const threadIdx = globalIdx % threadsPerBlock;
    const warp = Math.floor(threadIdx / 32);
    const indices = [];

    if (example === "grid-stride") {
      for (let i = globalIdx; i < dataSize; i += launchedThreads) indices.push(i);
    } else if (globalIdx < dataSize) {
      indices.push(globalIdx);
    }

    for (const index of indices) {
      if (example === "vector-add" || example === "residual-add") output[index] = inputA[index] + inputB[index];
      if (example === "grid-stride") output[index] = inputA[index] * 2;
      if (example === "bounds") output[index] = inputA[index];
      if (example === "indexing") output[index] = globalIdx;
    }

    return {
      blockIdx,
      threadIdx,
      globalIdx,
      warp,
      active: indices.length > 0,
      indices,
    };
  });

  return {
    ...normalized,
    launchedThreads,
    extraThreads: threads.filter((thread) => !thread.active).length,
    inputA,
    inputB,
    output,
    threads,
    metadata: EXAMPLES[example],
  };
}

export function stageState(simulation, step) {
  const safeStep = clampInteger(step, 0, 4);
  const labels = [
    "1. Allocate CPU data",
    "2. Copy inputs to device memory",
    "3. Launch the grid",
    "4. Execute active threads",
    "5. Copy results to the CPU",
  ];
  return {
    step: safeStep,
    label: labels[safeStep],
    hostReady: true,
    deviceReady: safeStep >= 1,
    gridVisible: safeStep >= 2,
    executed: safeStep >= 3,
    resultOnHost: safeStep >= 4,
    simulation,
  };
}

export { EXAMPLES };
