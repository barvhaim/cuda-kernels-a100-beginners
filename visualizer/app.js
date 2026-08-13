import { createSimulation, stageState } from "./emulator.js";

const $ = (selector) => document.querySelector(selector);
const controls = {
  example: $("#example"),
  blocks: $("#blocks"),
  threadsPerBlock: $("#threads"),
  dataSize: $("#data-size"),
};

const steps = [
  ["CPU creates the data", "The arrays begin in normal CPU memory."],
  ["Copy inputs to the GPU", "cudaMemcpy makes device copies of the input arrays."],
  ["Launch the CUDA grid", "The launch creates blocks. Each block contains threads."],
  ["Each thread does one job", "The selected thread computes its global index, checks the boundary, and updates its item."],
  ["Copy the answer to the CPU", "The completed output returns to CPU memory for checking or display."],
];

let step = 0;
let selectedGlobalIdx = 0;
let simulation;

function config() {
  return {
    example: controls.example.value,
    blocks: controls.blocks.value,
    threadsPerBlock: controls.threadsPerBlock.value,
    dataSize: controls.dataSize.value,
  };
}

function selectedThread() {
  return simulation.threads[selectedGlobalIdx] || simulation.threads[0];
}

function renderStep() {
  $("#step-number").textContent = `STEP ${step + 1} OF 5`;
  $("#step-title").textContent = steps[step][0];
  $("#step-explanation").textContent = steps[step][1];
  $("#pipeline").innerHTML = steps.map(([title], index) =>
    `<li class="${index === step ? "current" : index < step ? "done" : ""}"><b>${index + 1}</b><span>${title}</span></li>`
  ).join("");
  $("#play-step").textContent = step === 4 ? "Replay ↻" : "Next →";
  $("#previous-step").disabled = step === 0;
}

function renderFormula() {
  const thread = selectedThread();
  $("#formula-example").textContent = `${thread.blockIdx} × ${simulation.threadsPerBlock} + ${thread.threadIdx} = ${thread.globalIdx}`;
}

function renderExecutionMap() {
  const state = stageState(simulation, step);
  const thread = selectedThread();
  const targets = thread.indices.length ? thread.indices.map((index) => `item ${index}`).join(", ") : "no item: outside data";
  const inputs = simulation.example === "vector-add" ? "A  B" : "A";

  $("#machine-map").innerHTML = `
    <div class="scene-node ${step === 0 ? "focus" : ""}"><span class="icon cpu">CPU</span><b>Host memory</b><small>${inputs}</small></div>
    <div class="arrow ${step === 1 ? "moving" : ""}"><span>cudaMemcpy H→D</span><b>→</b></div>
    <div class="scene-node ${state.deviceReady ? "visible" : ""} ${step === 1 ? "focus" : ""}"><span class="icon gpu">GPU</span><b>Device memory</b><small>${inputs} → C</small></div>
    <div class="arrow ${step === 2 ? "moving" : ""}"><span>launch</span><b>→</b></div>
    <div class="grid-picture ${state.gridVisible ? "visible" : ""} ${step === 2 || step === 3 ? "focus" : ""}">
      <label>GRID</label>
      <div class="block-picture"><span>BLOCK ${thread.blockIdx}</span><div class="chosen-thread">THREAD ${thread.threadIdx}<strong>i = ${thread.globalIdx}</strong></div></div>
      <div class="target-arrow ${state.executed ? "moving-down" : ""}">↓</div>
      <div class="target-item ${state.executed ? "filled" : ""}">${targets}</div>
    </div>
    <div class="return-arrow ${step === 4 ? "moving" : ""}"><b>←</b><span>copy result</span></div>`;
}

function cells(values, visible, result = false) {
  const thread = selectedThread();
  return `<div class="cells">${values.map((value, index) => {
    const shown = visible && value !== null;
    const target = thread.indices.includes(index);
    return `<div class="cell ${shown && result ? "result" : ""} ${target ? "thread-target" : ""}" data-index="${index}">${shown ? value : "·"}</div>`;
  }).join("")}</div>`;
}

function renderMemory() {
  const state = stageState(simulation, step);
  const rows = [];
  rows.push(`<div class="memory-row"><b>CPU input A</b>${cells(simulation.inputA, true)}</div>`);
  if (simulation.example === "vector-add") rows.push(`<div class="memory-row"><b>CPU input B</b>${cells(simulation.inputB, true)}</div>`);
  if (state.deviceReady) {
    rows.push(`<div class="memory-divider">copied to GPU ↓</div>`);
    rows.push(`<div class="memory-row device-row"><b>GPU input A</b>${cells(simulation.inputA, true)}</div>`);
    if (simulation.example === "vector-add") rows.push(`<div class="memory-row device-row"><b>GPU input B</b>${cells(simulation.inputB, true)}</div>`);
  }
  if (state.executed) rows.push(`<div class="memory-row output-row"><b>GPU output C</b>${cells(simulation.output, true, true)}</div>`);
  if (state.resultOnHost) rows.push(`<div class="memory-row result-row"><b>CPU result</b>${cells(simulation.output, true, true)}</div>`);
  $("#memory-view").innerHTML = rows.join("");
}

function renderStats() {
  $("#stats").innerHTML = [
    [simulation.blocks, "blocks"],
    [simulation.threadsPerBlock, "threads / block"],
    [simulation.launchedThreads, "threads launched"],
    [simulation.extraThreads, "extra threads"],
  ].map(([value, label]) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`).join("");
}

function renderBlocks() {
  $("#blocks-view").innerHTML = Array.from({ length: simulation.blocks }, (_, blockIdx) => {
    const threads = simulation.threads.filter((thread) => thread.blockIdx === blockIdx);
    return `<article class="block"><header>BLOCK ${blockIdx}</header><div class="threads">${threads.map((thread) =>
      `<button class="thread ${thread.active ? "" : "idle"} ${thread.globalIdx === selectedGlobalIdx ? "selected" : ""}" data-thread="${thread.globalIdx}"><small>T${thread.threadIdx}</small><b>i=${thread.globalIdx}</b></button>`
    ).join("")}</div></article>`;
  }).join("");

  document.querySelectorAll("[data-thread]").forEach((button) => button.addEventListener("click", () => {
    selectedGlobalIdx = Number(button.dataset.thread);
    renderFormula();
    renderExecutionMap();
    renderMemory();
    renderBlocks();
    renderInspector();
  }));
}

function renderInspector() {
  const thread = selectedThread();
  const work = thread.indices.length ? thread.indices.map((i) => `data[${i}]`).join(", ") : "nothing: the bounds check stops it";
  $("#inspector").innerHTML = `<b>Selected thread:</b> block ${thread.blockIdx}, thread ${thread.threadIdx} → global index ${thread.globalIdx} → ${work}.`;
}

function renderCode() {
  $("#code-view").textContent = simulation.metadata.code.join("\n");
  $("#lesson-link").innerHTML = `Continue in <a href="../${simulation.metadata.lesson}">${simulation.metadata.lesson}</a>.`;
}

function renderAll() {
  simulation = createSimulation(config());
  selectedGlobalIdx = Math.min(selectedGlobalIdx, simulation.launchedThreads - 1);
  $("#blocks-value").textContent = simulation.blocks;
  $("#threads-value").textContent = simulation.threadsPerBlock;
  $("#data-value").textContent = simulation.dataSize;
  renderStep();
  renderFormula();
  renderExecutionMap();
  renderMemory();
  renderStats();
  renderBlocks();
  renderInspector();
  renderCode();
}

Object.values(controls).forEach((control) => control.addEventListener("input", () => {
  step = 0;
  selectedGlobalIdx = 0;
  renderAll();
}));
$("#play-step").addEventListener("click", () => { step = step === 4 ? 0 : step + 1; renderAll(); });
$("#previous-step").addEventListener("click", () => { step = Math.max(0, step - 1); renderAll(); });
$("#reset-step").addEventListener("click", () => { step = 0; selectedGlobalIdx = 0; renderAll(); });

renderAll();
