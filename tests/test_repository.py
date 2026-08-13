import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = {
    "00_device_query": "lessons/00_device_query.cu",
    "01_hello_kernel": "lessons/01_hello_kernel.cu",
    "02_one_block_index": "lessons/02_one_block_index.cu",
    "03_global_index": "lessons/03_global_index.cu",
    "04_bounds_check": "lessons/04_bounds_check.cu",
    "05_memory_roundtrip": "lessons/05_memory_roundtrip.cu",
    "06_vector_add": "lessons/06_vector_add.cu",
    "07_grid_stride": "lessons/07_grid_stride.cu",
    "08_reduction": "lessons/08_reduction.cu",
    "09_tiled_matmul": "lessons/09_tiled_matmul.cu",
}
LLM_EXAMPLES = {
    "llm_01_token_embedding": "llm_examples/01_token_embedding.cu",
    "llm_02_residual_add": "llm_examples/02_residual_add.cu",
    "llm_03_silu_activation": "llm_examples/03_silu_activation.cu",
    "llm_04_rmsnorm": "llm_examples/04_rmsnorm.cu",
    "llm_05_causal_mask": "llm_examples/05_causal_mask.cu",
    "llm_06_attention_softmax": "llm_examples/06_attention_softmax.cu",
    "llm_07_linear_projection": "llm_examples/07_linear_projection.cu",
    "llm_08_mini_transformer_step": "llm_examples/08_mini_transformer_step.cu",
}
NOTEBOOKS = [
    "00_indexing_cpu.ipynb",
    "01_cuda_basics_a100.ipynb",
    "01_vector_add_a100.ipynb",
    "02_memory_patterns_a100.ipynb",
    "03_profile_a100.ipynb",
    "04_llm_building_blocks.ipynb",
]


def notebook_source(filename):
    data = json.loads((ROOT / "notebooks" / filename).read_text())
    return "".join(
        "".join(cell["source"])
        for cell in data["cells"]
        if cell["cell_type"] == "code"
    )


class RepositoryContractTests(unittest.TestCase):
    def test_course_content_is_english_only(self):
        hebrew = re.compile(r"[\u0590-\u05FF]")
        bidi_controls = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
        learning_paths = [
            ROOT / "README.md",
            ROOT / "docs",
            ROOT / "exercises",
            ROOT / "solutions",
            ROOT / "notebooks",
        ]
        files = []
        for path in learning_paths:
            files.extend([path] if path.is_file() else path.rglob("*"))
        for path in files:
            if not path.is_file():
                continue
            text = path.read_text()
            self.assertIsNone(hebrew.search(text), f"Hebrew text remains in {path}")
            self.assertIsNone(bidi_controls.search(text), f"Bidi control remains in {path}")

    def test_all_cuda_targets_exist_and_are_built(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        for target, relative_source in {**LESSONS, **LLM_EXAMPLES}.items():
            self.assertTrue((ROOT / relative_source).is_file(), relative_source)
            self.assertIn(f"add_cuda_lesson({target} {relative_source})", cmake)

    def test_a100_architecture_is_explicit(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        self.assertRegex(cmake, r"CMAKE_CUDA_ARCHITECTURES\s+80")

    def test_beginner_sequence_is_single_concept_and_ordered(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        positions = [cmake.index(f"add_cuda_lesson({target} ") for target in LESSONS]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("hello_kernel<<<1, 1>>>", (ROOT / LESSONS["01_hello_kernel"]).read_text())
        self.assertIn("threadIdx.x", (ROOT / LESSONS["02_one_block_index"]).read_text())
        self.assertIn(
            "blockIdx.x * blockDim.x + threadIdx.x",
            (ROOT / LESSONS["03_global_index"]).read_text(),
        )

    def test_reductions_reject_non_power_of_two_block_sizes(self):
        for relative_source in (
            LESSONS["08_reduction"],
            LLM_EXAMPLES["llm_04_rmsnorm"],
            LLM_EXAMPLES["llm_06_attention_softmax"],
        ):
            source = (ROOT / relative_source).read_text()
            self.assertIn("static_assert", source, relative_source)
            self.assertIn("threads & (threads - 1)", source, relative_source)

    def test_cuda_examples_check_runtime_calls(self):
        for relative_source in (*LESSONS.values(), *LLM_EXAMPLES.values()):
            text = (ROOT / relative_source).read_text()
            self.assertTrue(
                "CUDA_CHECK" in text or "check_kernel()" in text,
                relative_source,
            )

    def test_array_kernels_have_boundary_guards(self):
        expectations = {
            LESSONS["04_bounds_check"]: "if (i < n)",
            LESSONS["05_memory_roundtrip"]: "if (i < n)",
            LESSONS["06_vector_add"]: "if (i < n)",
            LESSONS["07_grid_stride"]: "i < n",
            LESSONS["08_reduction"]: "if (i < n)",
            LESSONS["09_tiled_matmul"]: "if (row < n && col < n)",
            LLM_EXAMPLES["llm_02_residual_add"]: "if (i < elements)",
            LLM_EXAMPLES["llm_05_causal_mask"]: "query < sequence_length && key < sequence_length",
            LLM_EXAMPLES["llm_07_linear_projection"]: "if (out < output_size)",
        }
        for relative_source, guard in expectations.items():
            self.assertIn(guard, (ROOT / relative_source).read_text(), relative_source)

    def test_llm_examples_cover_core_building_blocks(self):
        guide = (ROOT / "docs" / "07-llm-kernel-map.md").read_text().lower()
        for term in ("embedding", "residual", "silu", "rmsnorm", "causal", "softmax", "projection"):
            self.assertIn(term, guide)
        pipeline = (ROOT / LLM_EXAMPLES["llm_08_mini_transformer_step"]).read_text()
        for kernel in ("embedding_lookup<<<", "residual_add<<<", "rmsnorm<<<", "project<<<"):
            self.assertIn(kernel, pipeline)

    def test_embedding_validates_token_ids(self):
        source = (ROOT / LLM_EXAMPLES["llm_01_token_embedding"]).read_text()
        self.assertIn("token_id >= 0 && token_id < vocab_size", source)
        self.assertIn("Token ID out of vocabulary range", source)

    def test_mini_pipeline_compares_logits_to_cpu_reference(self):
        source = (ROOT / LLM_EXAMPLES["llm_08_mini_transformer_step"]).read_text()
        self.assertIn("std::vector<float> expected", source)
        self.assertIn("std::fabs(logits[token] - expected[token])", source)

    def test_one_block_llm_examples_state_shape_limits(self):
        embedding = (ROOT / LLM_EXAMPLES["llm_01_token_embedding"]).read_text()
        rmsnorm = (ROOT / LLM_EXAMPLES["llm_04_rmsnorm"]).read_text()
        softmax = (ROOT / LLM_EXAMPLES["llm_06_attention_softmax"]).read_text()
        pipeline = (ROOT / LLM_EXAMPLES["llm_08_mini_transformer_step"]).read_text()
        self.assertIn("hidden_size <= 1024", embedding)
        self.assertIn("threads >= hidden_size", rmsnorm)
        self.assertIn("threads >= n", softmax)
        self.assertIn("hidden_size <= 1024 && vocab_size <= 1024", pipeline)

    def test_attention_softmax_is_numerically_stable(self):
        source = (ROOT / LLM_EXAMPLES["llm_06_attention_softmax"]).read_text()
        self.assertIn("scores[tid] - maxima[0]", source)
        self.assertIn("1000.0f", source)
        self.assertIn("std::isfinite", source)
        self.assertIn("probabilities[i] < 0.0f", source)
        self.assertIn("probabilities[i] - expected[i]", source)

    def test_silu_and_causal_mask_teach_stable_numerics(self):
        silu = (ROOT / LLM_EXAMPLES["llm_03_silu_activation"]).read_text()
        mask = (ROOT / LLM_EXAMPLES["llm_05_causal_mask"]).read_text()
        self.assertIn("x >= 0.0f", silu)
        self.assertIn("-100.0f", silu)
        self.assertIn("-INFINITY", mask)

    def test_readme_local_links_exist(self):
        readme = (ROOT / "README.md").read_text()
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
        local_links = [link.split("#", 1)[0] for link in links if "://" not in link]
        for link in local_links:
            self.assertTrue((ROOT / link).exists(), link)

    def test_run_script_covers_every_target_and_track(self):
        script = (ROOT / "scripts" / "run_all.sh").read_text()
        for target in (*LESSONS, *LLM_EXAMPLES):
            self.assertIn(target, script)
        for track in ("foundations", "patterns", "basics", "llm", "all"):
            self.assertIn(track, script)

    def test_notebooks_are_valid_and_linked(self):
        readme = (ROOT / "README.md").read_text()
        for filename in NOTEBOOKS:
            path = ROOT / "notebooks" / filename
            data = json.loads(path.read_text())
            self.assertEqual(data["nbformat"], 4)
            self.assertGreaterEqual(len(data["cells"]), 3)
            self.assertTrue(any(cell["cell_type"] == "code" for cell in data["cells"]))
            self.assertIn(f"notebooks/{filename}", readme)

    def test_a100_notebooks_do_not_contain_fake_outputs(self):
        for filename in NOTEBOOKS[1:]:
            data = json.loads((ROOT / "notebooks" / filename).read_text())
            for cell in data["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs"), [], filename)
                    self.assertIsNone(cell.get("execution_count"), filename)

    def test_a100_notebooks_find_repo_root_upward(self):
        for filename in NOTEBOOKS[1:]:
            source = notebook_source(filename)
            self.assertIn("for candidate in (start, *start.parents)", source)
            self.assertIn("raise RuntimeError", source)

    def test_first_a100_notebook_validates_selected_device(self):
        source = notebook_source("01_vector_add_a100.ipynb")
        self.assertIn('["nvidia-smi", "-L"]', source)
        self.assertIn('device_zero = device.stdout.split("Device 1:", 1)[0]', source)
        self.assertIn('"A100" in device_zero', source)
        self.assertIn('"compute capability: 8.0" in device_zero', source)
        self.assertIn('"00_device_query", "06_vector_add"', source)

    def test_basics_notebook_runs_the_foundation_targets_in_order(self):
        source = notebook_source("01_cuda_basics_a100.ipynb")
        positions = [source.index(target) for target in list(LESSONS)[:7]]
        self.assertEqual(positions, sorted(positions))

    def test_llm_notebook_has_cpu_reference_and_all_gpu_targets(self):
        source = notebook_source("04_llm_building_blocks.ipynb")
        self.assertIn("embeddings = [", source)
        self.assertIn("probabilities =", source)
        for target in LLM_EXAMPLES:
            self.assertIn(target, source)
        self.assertIn('"A100" in device_zero', source)
        self.assertIn('"compute capability: 8.0" in device_zero', source)
        self.assertIn("score + mask[query][key]", source)
        self.assertIn("probabilities[2:] == [0.0, 0.0]", source)
        self.assertIn('line.startswith("LOGITS ")', source)
        self.assertIn("zip(cuda_logits, expected_logits)", source)

    def test_cpu_visualizer_is_a_zero_dependency_learning_path(self):
        visualizer = ROOT / "visualizer"
        expected_files = ("index.html", "styles.css", "app.js", "emulator.js")
        for filename in expected_files:
            self.assertTrue((visualizer / filename).is_file(), filename)

        html = (visualizer / "index.html").read_text()
        app = (visualizer / "app.js").read_text()
        emulator = (visualizer / "emulator.js").read_text()
        readme = (ROOT / "README.md").read_text()

        self.assertIn("CPU → GPU execution map", html)
        self.assertIn('id="story-canvas"', html)
        self.assertIn('id="step-title"', html)
        self.assertIn("How does a CUDA thread", html)
        self.assertIn("choose its data?", html)
        self.assertIn('id="mapping-view"', html)
        self.assertIn('id="inline-code"', html)
        self.assertIn('class="lesson-columns"', html)
        self.assertIn("Code running beside the picture", html)
        self.assertIn("Try changing the launch", html)
        self.assertNotIn('class="execution-map"', html)
        self.assertNotIn('class="journey"', html)
        for example in ("indexing", "bounds", "vector-add", "grid-stride"):
            self.assertIn(f'value="{example}"', html)
        self.assertIn("blockIdx.x * blockDim.x + threadIdx.x", html)
        self.assertIn("CPU execution-model emulator", html)
        self.assertIn("not a performance simulator", html)
        self.assertIn("createSimulation", app)
        self.assertIn("GPU input A", app)
        self.assertIn("GPU input B", app)
        self.assertIn("GPU output C", app)
        self.assertIn("renderExecutionMap", app)
        self.assertIn("renderMapping", app)
        self.assertIn("renderInlineCode", app)
        self.assertIn("const activeLine", app)
        self.assertIn('? "current" : ""', app)
        self.assertIn("outside array", app)
        self.assertIn("thread-target", app)
        self.assertIn("cudaMemcpy H→D", app)
        self.assertIn("export function createSimulation", emulator)
        self.assertIn("visualizer/index.html", readme)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

        pages_entry = (ROOT / "index.html").read_text()
        self.assertIn('url=visualizer/', pages_entry)
        self.assertIn('href="visualizer/"', pages_entry)

    def test_ci_is_honest_about_not_building_cuda(self):
        workflow = (ROOT / ".github" / "workflows" / "checks.yml").read_text()
        self.assertNotIn("nvidia-smi", workflow)
        self.assertNotIn("make run", workflow)
        self.assertIn("CPU-only", workflow)


if __name__ == "__main__":
    unittest.main()
