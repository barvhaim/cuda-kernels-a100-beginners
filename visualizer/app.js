import { createSimulation, stageState } from "./emulator.js";

const $ = (selector) => document.querySelector(selector);
const controls = {
  example: $("#example"),
  blocks: $("#blocks"),
  threadsPerBlock: $("#threads"),
  dataSize: $("#data-size"),
};

const steps = [
  ["Launch", "CUDA creates all logical threads in the blocks below."],
  ["Index", "Every thread calculates its own global index."],
  ["Check", "A thread works only when its index is smaller than the array length."],
  ["Update", "Active threads map to array items. Extra threads stop at the boundary."],
];

let step = 0;
let selectedGlobalIdx = 0;
let simulation;
let timelineTimer = null;
let isPlaying = false;

function loadUrlState() {
  const params = new URLSearchParams(window.location.search);
  if (["indexing", "bounds", "vector-add", "grid-stride"].includes(params.get("example"))) controls.example.value = params.get("example");
  if (params.has("blocks")) controls.blocks.value = params.get("blocks");
  if (params.has("threads")) controls.threadsPerBlock.value = params.get("threads");
  if (params.has("n")) controls.dataSize.value = params.get("n");
  step = Math.max(0, Math.min(3, Number(params.get("frame")) || 0));
  selectedGlobalIdx = Math.max(0, Number(params.get("thread")) || 0);
}

function syncUrlState() {
  const params = new URLSearchParams({
    example: controls.example.value,
    blocks: controls.blocks.value,
    threads: controls.threadsPerBlock.value,
    n: controls.dataSize.value,
    frame: String(step),
    thread: String(selectedGlobalIdx),
  });
  history.replaceState(null, "", `${window.location.pathname}?${params}`);
}

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

function inlineCodeLines() {
  const work = {
    indexing: 'printf("global=%d\\n", i);',
    bounds: "data[i] = data[i];",
    "vector-add": "c[i] = a[i] + b[i];",
    "grid-stride": "output[i] = input[i] * 2;",
  }[simulation.example];
  if (simulation.example === "grid-stride") {
    return [
      `kernel<<<${simulation.blocks}, ${simulation.threadsPerBlock}>>>();`,
      "int i = blockIdx.x * blockDim.x + threadIdx.x;",
      "for (; i < n; i += blockDim.x * gridDim.x) {",
      `  ${work}`,
      "}",
    ];
  }
  return [
    `kernel<<<${simulation.blocks}, ${simulation.threadsPerBlock}>>>();`,
    "int i = blockIdx.x * blockDim.x + threadIdx.x;",
    "if (i < n) {",
    `  ${work}`,
    "}",
  ];
}

function renderInlineCode() {
  const activeLine = [0, 1, 2, 3][step];
  const explanations = [
    "The launch syntax creates the blocks and threads shown on the left.",
    "Every thread runs this same line, but its blockIdx and threadIdx are different.",
    "The boundary check prevents indices outside the array from accessing memory.",
    "Only a valid thread reaches the work line and updates its mapped item.",
  ];
  $("#inline-code").innerHTML = inlineCodeLines().map((line, index) => {
    const escaped = line.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    return `<div class="code-line ${index === activeLine ? "current" : ""}"><span>${index + 1}</span><code>${escaped}</code></div>`;
  }).join("");
  $("#code-explanation").textContent = explanations[step];
}

function renderStep() {
  $("#step-number").textContent = `FRAME ${step + 1} OF 4`;
  $("#step-title").textContent = steps[step][0];
  $("#step-explanation").textContent = steps[step][1];
  $("#pipeline").innerHTML = steps.map(([title], index) =>
    `<li class="${index === step ? "current" : index < step ? "done" : ""}" data-frame="${index}"><b>${index + 1}</b><span>${title}</span></li>`
  ).join("");
  document.querySelectorAll("[data-frame]").forEach((item) => item.addEventListener("click", () => {
    stopTimeline();
    step = Number(item.dataset.frame);
    renderAll();
  }));
  $("#play-step").textContent = step === 3 ? "Replay" : "Next frame";
  $("#previous-step").disabled = step === 0;
  $("#frame-scrubber").value = step;
  $("#timeline-time").textContent = `00:0${step + 1} / 00:04`;
  $("#play-timeline").textContent = isPlaying ? "Ⅱ" : "▶";
  $("#play-timeline").setAttribute("aria-label", isPlaying ? "Pause timeline" : "Play timeline");
  document.body.dataset.frame = String(step);
}

function stopTimeline() {
  if (timelineTimer) window.clearInterval(timelineTimer);
  timelineTimer = null;
  isPlaying = false;
}

function playTimeline() {
  if (isPlaying) {
    stopTimeline();
    renderAll();
    return;
  }
  if (step === 3) step = 0;
  isPlaying = true;
  renderAll();
  timelineTimer = window.setInterval(() => {
    if (step === 3) {
      stopTimeline();
      renderAll();
      return;
    }
    step += 1;
    renderAll();
  }, 1250);
}

function renderFormula() {
  const thread = selectedThread();
  const valid = thread.globalIdx < simulation.dataSize;
  $("#selected-plain").textContent = `Block ${thread.blockIdx}, Thread ${thread.threadIdx}`;
  $("#formula-example").textContent = `(${thread.blockIdx} × ${simulation.threadsPerBlock}) + ${thread.threadIdx} = ${thread.globalIdx}`;
  $("#bounds-result").innerHTML = valid
    ? `<b>${thread.globalIdx} &lt; ${simulation.dataSize}</b> → valid → works on item ${thread.globalIdx}`
    : `<b>${thread.globalIdx} &lt; ${simulation.dataSize}</b> → false → does nothing`;
}

function renderMapping() {
  const selected = selectedThread();
  $("#launch-summary").textContent = `${simulation.blocks} blocks × ${simulation.threadsPerBlock} threads = ${simulation.launchedThreads} threads for ${simulation.dataSize} items`;
  $("#launch-consequence").textContent = simulation.extraThreads
    ? `${simulation.extraThreads} extra thread${simulation.extraThreads === 1 ? "" : "s"} fall outside the array and do no work.`
    : "Every launched thread maps to data in this example.";

  $("#mapping-view").innerHTML = Array.from({ length: simulation.blocks }, (_, blockIdx) => {
    const threads = simulation.threads.filter((thread) => thread.blockIdx === blockIdx);
    return `<article class="mapping-block"><header>BLOCK ${blockIdx}<small>blockDim.x = ${simulation.threadsPerBlock}</small></header><div class="mapping-threads">${threads.map((thread) => {
      const target = thread.indices.length ? thread.indices.map((index) => `A[${index}]`).join(", ") : "outside array";
      const isSelected = thread.globalIdx === selected.globalIdx;
      const frameState = step === 0 ? "spawned" : step === 1 ? "indexed" : step === 2 ? (thread.active ? "accepted" : "rejected") : (thread.active ? "committed" : "rejected");
      return `<button class="mapping-lane ${thread.active ? "active" : "extra"} ${isSelected ? "selected" : ""} ${frameState}" data-thread="${thread.globalIdx}">
        <span class="lane-thread"><i class="thread-spark"></i>T${thread.threadIdx}<small>${step === 0 ? "ready" : `i=${thread.globalIdx}`}</small></span>
        <span class="lane-arrow">${step < 2 ? "↓" : thread.active ? "↓" : "×"}</span>
        <span class="lane-target">${step === 0 ? "waiting" : step === 1 ? `index ${thread.globalIdx}` : target}</span>
      </button>`;
    }).join("")}</div></article>`;
  }).join("");

  document.querySelectorAll(".mapping-lane").forEach((button) => button.addEventListener("click", () => {
    stopTimeline();
    selectedGlobalIdx = Number(button.dataset.thread);
    renderAll();
  }));
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
  $(".cinema-bar > span").textContent = `${simulation.metadata.lesson.split("/").pop()} · execution trace`;
}

function renderAll() {
  simulation = createSimulation(config());
  selectedGlobalIdx = Math.min(selectedGlobalIdx, simulation.launchedThreads - 1);
  $("#blocks-value").textContent = simulation.blocks;
  $("#threads-value").textContent = simulation.threadsPerBlock;
  $("#data-value").textContent = simulation.dataSize;
  renderStep();
  renderInlineCode();
  renderMapping();
  renderFormula();
  renderExecutionMap();
  renderMemory();
  renderStats();
  renderBlocks();
  renderInspector();
  renderCode();
  syncUrlState();
}

Object.values(controls).forEach((control) => control.addEventListener("input", () => {
  stopTimeline();
  step = 0;
  selectedGlobalIdx = 0;
  renderAll();
}));
$("#play-step").addEventListener("click", () => { stopTimeline(); step = step === 3 ? 0 : step + 1; renderAll(); });
$("#previous-step").addEventListener("click", () => { stopTimeline(); step = Math.max(0, step - 1); renderAll(); });
$("#reset-step").addEventListener("click", () => { stopTimeline(); step = 0; selectedGlobalIdx = 0; renderAll(); });
$("#play-timeline").addEventListener("click", playTimeline);
$("#frame-scrubber").addEventListener("input", (event) => { stopTimeline(); step = Number(event.target.value); renderAll(); });
$("#copy-lesson-link").addEventListener("click", async () => {
  syncUrlState();
  const button = $("#copy-lesson-link");
  try {
    await navigator.clipboard.writeText(window.location.href);
    button.textContent = "✓ Link copied";
  } catch {
    window.prompt("Copy this lesson link", window.location.href);
  }
  window.setTimeout(() => { button.textContent = "⌁ Copy lesson link"; }, 1600);
});

loadUrlState();
renderAll();
