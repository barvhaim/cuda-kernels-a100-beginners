import { createSimulation, stageState } from "./emulator.js";

const $ = (selector) => document.querySelector(selector);
const controls = {
  example: $("#example"),
  blocks: $("#blocks"),
  threadsPerBlock: $("#threads"),
  dataSize: $("#data-size"),
};

const stageCopy = [
  "The CPU creates ordinary input arrays. Nothing has reached a GPU yet.",
  "cudaMemcpy conceptually copies the inputs from host memory to device memory.",
  "The launch creates every logical thread. Each one computes the same formula with different IDs.",
  "Threads passing the bounds check update their assigned elements. Extra threads do no array work.",
  "The result is copied back to host memory where CPU code can print or verify it.",
];

let step = 0;
let selectedGlobalIdx = 0;
let playTimer = null;
let simulation;

function currentConfig() {
  return {
    example: controls.example.value,
    blocks: controls.blocks.value,
    threadsPerBlock: controls.threadsPerBlock.value,
    dataSize: controls.dataSize.value,
  };
}

function renderPipeline() {
  const state = stageState(simulation, step);
  const labels = ["CPU inputs", "Device inputs", "Kernel launch", "Thread work", "CPU result"];
  $("#pipeline").innerHTML = labels.map((label, index) => {
    const status = index < state.step ? "complete" : index === state.step ? "current" : "";
    return `<li class="${status}"><span class="number">STEP ${index + 1}</span>${label}</li>`;
  }).join("");
  $("#step-explanation").innerHTML = `<strong>${state.label}</strong><br>${stageCopy[state.step]}`;
  $("#play-step").textContent = step === 4 ? "Replay" : "Next step";
}

function renderStats() {
  const active = simulation.threads.filter((thread) => thread.active).length;
  $("#stats").innerHTML = [
    [simulation.blocks, "Blocks in Grid"],
    [simulation.threadsPerBlock, "Threads per Block"],
    [simulation.launchedThreads, "Threads launched"],
    [simulation.extraThreads, "Extra Threads"],
  ].map(([value, label]) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`).join("");
  $("#stats").setAttribute("aria-label", `${active} active logical threads`);
}

function renderBlocks() {
  $("#blocks-view").innerHTML = Array.from({ length: simulation.blocks }, (_, blockIdx) => {
    const threads = simulation.threads.filter((thread) => thread.blockIdx === blockIdx);
    return `<article class="block">
      <div class="block-header"><span>BLOCK ${blockIdx}</span><span>blockDim.x = ${simulation.threadsPerBlock}</span></div>
      <div class="threads">${threads.map((thread) => `
        <button class="thread ${thread.active ? "" : "idle"} ${thread.globalIdx === selectedGlobalIdx ? "selected" : ""}"
          data-thread="${thread.globalIdx}" aria-label="Block ${thread.blockIdx}, thread ${thread.threadIdx}, global index ${thread.globalIdx}">
          <small>threadIdx ${thread.threadIdx}</small><b>i = ${thread.globalIdx}</b><small>warp ${thread.warp}</small>
        </button>`).join("")}</div>
    </article>`;
  }).join("");

  document.querySelectorAll("[data-thread]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedGlobalIdx = Number(button.dataset.thread);
      renderBlocks();
      renderInspector();
    });
  });
}

function renderInspector() {
  const thread = simulation.threads[selectedGlobalIdx] || simulation.threads[0];
  const math = `${thread.blockIdx} × ${simulation.threadsPerBlock} + ${thread.threadIdx} = ${thread.globalIdx}`;
  const work = thread.indices.length
    ? `works on ${thread.indices.map((index) => `data[${index}]`).join(", ")}`
    : `fails i &lt; n (${thread.globalIdx} &lt; ${simulation.dataSize}) and does no array work`;
  $("#inspector").innerHTML = `<strong>Thread inspector</strong> · <code>${math}</code><br>
    Block <b>${thread.blockIdx}</b>, thread <b>${thread.threadIdx}</b>, global index <b>${thread.globalIdx}</b>, warp <b>${thread.warp}</b>: ${work}.`;
  $("#formula-example").textContent = `Selected: ${math}`;
}

function cells(values, visible, result = false) {
  return `<div class="cells">${values.map((value, index) => {
    const shown = visible && value !== null;
    return `<div class="cell ${shown && result ? "result" : ""} ${shown ? "" : "pending"}" data-index="${index}">${shown ? value : "·"}</div>`;
  }).join("")}</div>`;
}

function renderMemory() {
  const state = stageState(simulation, step);
  const rows = [];
  rows.push(`<div class="memory-row"><div class="memory-label">HOST · input A</div>${cells(simulation.inputA, state.hostReady)}</div>`);
  if (simulation.example === "vector-add") {
    rows.push(`<div class="memory-row"><div class="memory-label">HOST · input B</div>${cells(simulation.inputB, state.hostReady)}</div>`);
  }
  rows.push(`<div class="memory-row"><div class="memory-label">DEVICE · input</div>${cells(simulation.inputA, state.deviceReady)}</div>`);
  rows.push(`<div class="memory-row"><div class="memory-label">DEVICE · output</div>${cells(simulation.output, state.executed, true)}</div>`);
  rows.push(`<div class="memory-row"><div class="memory-label">HOST · result</div>${cells(simulation.output, state.resultOnHost, true)}</div>`);
  $("#memory-view").innerHTML = rows.join("");
}

function renderCode() {
  $("#code-view").textContent = simulation.metadata.code.join("\n");
  $("#lesson-link").innerHTML = `Continue in <a href="../${simulation.metadata.lesson}">${simulation.metadata.lesson}</a>.`;
}

function render() {
  simulation = createSimulation(currentConfig());
  selectedGlobalIdx = Math.min(selectedGlobalIdx, simulation.launchedThreads - 1);
  $("#blocks-value").textContent = simulation.blocks;
  $("#threads-value").textContent = simulation.threadsPerBlock;
  $("#data-value").textContent = simulation.dataSize;
  renderPipeline();
  renderStats();
  renderBlocks();
  renderInspector();
  renderMemory();
  renderCode();
}

Object.values(controls).forEach((control) => control.addEventListener("input", () => {
  step = 0;
  selectedGlobalIdx = 0;
  render();
}));

$("#play-step").addEventListener("click", () => {
  step = step === 4 ? 0 : step + 1;
  renderPipeline();
  renderMemory();
});
$("#previous-step").addEventListener("click", () => {
  step = Math.max(0, step - 1);
  renderPipeline();
  renderMemory();
});
$("#reset-step").addEventListener("click", () => {
  clearInterval(playTimer);
  step = 0;
  renderPipeline();
  renderMemory();
});

render();
