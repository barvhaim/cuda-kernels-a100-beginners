import assert from "node:assert/strict";
import { createSimulation, normalizeConfig, stageState } from "../visualizer/emulator.js";

const bounds = createSimulation({ example: "bounds", blocks: 2, threadsPerBlock: 4, dataSize: 6 });
assert.equal(bounds.launchedThreads, 8);
assert.equal(bounds.extraThreads, 2);
assert.deepEqual(bounds.threads.map((thread) => thread.globalIdx), [0, 1, 2, 3, 4, 5, 6, 7]);
assert.deepEqual(bounds.output, [1, 2, 3, 4, 5, 6]);
assert.equal(bounds.threads[7].active, false);

const vector = createSimulation({ example: "vector-add", blocks: 2, threadsPerBlock: 3, dataSize: 5 });
assert.deepEqual(vector.output, [11, 22, 33, 44, 55]);
assert.equal(vector.extraThreads, 1);

const stride = createSimulation({ example: "grid-stride", blocks: 1, threadsPerBlock: 3, dataSize: 8 });
assert.deepEqual(stride.threads[0].indices, [0, 3, 6]);
assert.deepEqual(stride.threads[1].indices, [1, 4, 7]);
assert.deepEqual(stride.threads[2].indices, [2, 5]);
assert.deepEqual(stride.output, [2, 4, 6, 8, 10, 12, 14, 16]);

assert.deepEqual(normalizeConfig({ blocks: 99, threadsPerBlock: 0, dataSize: -4 }), {
  example: "bounds",
  blocks: 8,
  threadsPerBlock: 1,
  dataSize: 1,
});
assert.equal(stageState(bounds, 99).step, 4);
assert.equal(stageState(bounds, 3).executed, true);

console.log("PASS CPU emulator: indexing, bounds, vector add, grid stride, and stage states");
