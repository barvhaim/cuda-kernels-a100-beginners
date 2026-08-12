import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = [
    "00_device_query",
    "01_vector_add",
    "02_grid_stride",
    "03_reduction",
    "04_tiled_matmul",
]
NOTEBOOKS = [
    "00_indexing_cpu.ipynb",
    "01_vector_add_a100.ipynb",
    "02_memory_patterns_a100.ipynb",
    "03_profile_a100.ipynb",
]


class RepositoryContractTests(unittest.TestCase):
    def test_all_lessons_exist_and_are_built(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        for lesson in LESSONS:
            source = ROOT / "lessons" / f"{lesson}.cu"
            self.assertTrue(source.is_file(), source)
            self.assertIn(f"add_cuda_lesson({lesson} lessons/{lesson}.cu)", cmake)

    def test_a100_architecture_is_explicit(self):
        cmake = (ROOT / "CMakeLists.txt").read_text()
        self.assertRegex(cmake, r"CMAKE_CUDA_ARCHITECTURES\s+80")

    def test_cuda_examples_check_runtime_calls(self):
        for source in (ROOT / "lessons").glob("*.cu"):
            text = source.read_text()
            self.assertIn("CUDA_CHECK", text, source)

    def test_array_kernels_have_boundary_guards(self):
        expectations = {
            "01_vector_add.cu": "if (i < n)",
            "02_grid_stride.cu": "i < n",
            "03_reduction.cu": "if (i < n)",
            "04_tiled_matmul.cu": "if (row < n && col < n)",
        }
        for filename, guard in expectations.items():
            self.assertIn(guard, (ROOT / "lessons" / filename).read_text())

    def test_readme_local_links_exist(self):
        readme = (ROOT / "README.md").read_text()
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
        local_links = [link.split("#", 1)[0] for link in links if "://" not in link]
        for link in local_links:
            self.assertTrue((ROOT / link).exists(), link)

    def test_run_script_covers_every_lesson(self):
        script = (ROOT / "scripts" / "run_all.sh").read_text()
        for lesson in LESSONS:
            self.assertIn(lesson, script)

    def test_notebooks_are_valid_and_linked(self):
        import json

        readme = (ROOT / "README.md").read_text()
        for filename in NOTEBOOKS:
            path = ROOT / "notebooks" / filename
            data = json.loads(path.read_text())
            self.assertEqual(data["nbformat"], 4)
            self.assertGreaterEqual(len(data["cells"]), 3)
            self.assertTrue(any(cell["cell_type"] == "code" for cell in data["cells"]))
            self.assertIn(f"notebooks/{filename}", readme)

    def test_a100_notebooks_do_not_contain_fake_outputs(self):
        import json

        for filename in NOTEBOOKS[1:]:
            data = json.loads((ROOT / "notebooks" / filename).read_text())
            for cell in data["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs"), [], filename)
                    self.assertIsNone(cell.get("execution_count"), filename)

    def test_docs_do_not_claim_ci_runs_on_a100(self):
        workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text()
        self.assertNotIn("nvidia-smi", workflow)
        self.assertNotIn("make run", workflow)


if __name__ == "__main__":
    unittest.main()
