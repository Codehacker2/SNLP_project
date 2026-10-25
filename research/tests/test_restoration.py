"""Verify full notebook preservation and safe research execution without models.

Run: python -m unittest discover -s research/tests -v
Only the standard library is used; original experiments are never executed.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"research_{name}", RESEARCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_script("export_notebook")
runner = load_script("run")


class RestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(exporter.NOTEBOOK.read_bytes())
        cls.manifest = json.loads((RESEARCH / "source_map.json").read_text(encoding="utf-8"))
        cls.cache = RESEARCH / "cache"
        cls.cache.mkdir(exist_ok=True)

    def test_notebook_checksum_and_every_cell_in_original_order(self):
        self.assertEqual(self.manifest["notebook_sha256"], hashlib.sha256(exporter.NOTEBOOK.read_bytes()).hexdigest())
        self.assertEqual([entry["cell"] for entry in self.manifest["cells"]],
                         list(range(len(self.notebook["cells"]))))
        self.assertEqual(self.manifest["counts"]["code_cells"],
                         sum(cell["cell_type"] == "code" for cell in self.notebook["cells"]))

    def test_each_python_statement_and_source_line_is_preserved(self):
        for entry in self.manifest["cells"]:
            source = "".join(self.notebook["cells"][entry["cell"]]["source"])
            with self.subTest(cell=entry["cell"]):
                self.assertEqual(entry["source_sha256"], exporter.digest(source))
                if entry["type"] != "code" or "resource" in entry:
                    continue
                lines = (RESEARCH / entry["file"]).read_text(encoding="utf-8").splitlines()
                restored = "\n".join(lines[entry["start_line"] - 1:entry["end_line"]])
                self.assertEqual(restored, "\n".join(source.splitlines()))
                self.assertEqual(ast.dump(ast.parse(restored)), ast.dump(ast.parse(source)))

    def test_writefile_payloads_are_exact_including_historical_versions(self):
        versions = {}
        for entry in self.manifest["cells"]:
            if "resource" not in entry:
                continue
            source = "".join(self.notebook["cells"][entry["cell"]]["source"])
            body = source.partition("\n")[2]
            with self.subTest(cell=entry["cell"]):
                self.assertEqual((RESEARCH / entry["resource"]).read_text(encoding="utf-8"), body)
                self.assertEqual(entry["payload_sha256"], exporter.digest(body))
            versions.setdefault(entry["target"], []).append(entry["resource"])
        self.assertGreater(len(versions["src/exam_digitizer/api/app.py"]), 1)
        self.assertGreater(len(versions["README.md"]), 1)

    def test_string_embedded_modules_and_tests_have_readable_exact_snapshots(self):
        files = 0
        for entry in self.manifest["cells"]:
            if "embedded_files" not in entry:
                continue
            source = "".join(self.notebook["cells"][entry["cell"]]["source"])
            dictionaries = {
                target.id: ast.literal_eval(node.value)
                for node in ast.parse(source).body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name) and target.id in {"_SMALL", "_TESTS"}
            }
            for payload in entry["embedded_files"]:
                files += 1
                body = dictionaries[payload["variable"]][payload["target"]]
                with self.subTest(resource=payload["resource"]):
                    self.assertEqual((RESEARCH / payload["resource"]).read_text(encoding="utf-8"), body)
                    self.assertEqual(payload["payload_sha256"], exporter.digest(body))
        self.assertEqual(files, self.manifest["counts"]["embedded_files"])
        self.assertGreater(files, 0)

    def test_every_generated_file_is_reproducible_with_no_missing_or_stale_files(self):
        expected = exporter.build_export()
        self.assertEqual(exporter.generated_paths(), set(expected))
        for name, content in expected.items():
            with self.subTest(file=name):
                self.assertEqual((RESEARCH / name).read_text(encoding="utf-8"), content)

    def test_every_python_snapshot_and_stage_compiles_without_execution(self):
        for directory in ("stages", "file_cells"):
            for path in (RESEARCH / directory).rglob("*.py"):
                with self.subTest(file=path):
                    compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)

    def test_exporter_never_executes_notebook_cells(self):
        notebook = copy.deepcopy(self.notebook)
        notebook["cells"][2]["source"] = ["raise AssertionError('This cell must not execute')"]
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            path = Path(directory) / "source.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            output = exporter.build_export(path)
            self.assertIn("raise AssertionError", output["stages/00_setup.py"])

    def test_listing_and_verification_need_no_third_party_dependencies_or_outputs(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            for script, arguments in (("run.py", []), ("export_notebook.py", ["--check"])):
                result = subprocess.run([sys.executable, "-S", str(RESEARCH / script), *arguments],
                                        cwd=directory, capture_output=True, text=True, check=True)
                self.assertTrue(result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_stages_share_function_globals_and_do_not_inherit_future_annotations(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            folder = Path(directory)
            (folder / "first.py").write_text("value = 1\ndef read() -> int:\n    return value\n", encoding="utf-8")
            (folder / "second.py").write_text("value = 2\nresult = read()\n", encoding="utf-8")
            stages = [{"name": name, "file": name + ".py", "description": "test"} for name in ("first", "second")]
            namespace = {"__name__": "__main__"}
            with patch.object(runner, "RESEARCH", folder), contextlib.redirect_stdout(io.StringIO()):
                runner.run_stages(stages, namespace)
            self.assertEqual(namespace["result"], 2)
            self.assertIs(namespace["read"].__annotations__["return"], int)

    def test_exception_keeps_script_line_and_does_not_execute_later_stages(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            folder = Path(directory)
            (folder / "failed.py").write_text("raise ValueError('original failure')\n", encoding="utf-8")
            (folder / "next.py").write_text("should_not_run = True\n", encoding="utf-8")
            stages = [{"name": name, "file": name + ".py", "description": "test"} for name in ("failed", "next")]
            namespace = {}
            errors = io.StringIO()
            with patch.object(runner, "RESEARCH", folder), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
                try:
                    runner.run_stages(stages, namespace)
                except ValueError as exc:
                    trace = exc.__traceback__
                    while trace.tb_next:
                        trace = trace.tb_next
                    self.assertEqual(Path(trace.tb_frame.f_code.co_filename), folder / "failed.py")
                    self.assertEqual(trace.tb_lineno, 1)
                else:
                    self.fail("The original exception must propagate")
            self.assertNotIn("should_not_run", namespace)
            self.assertIn("Failed in failed", errors.getvalue())

    def test_run_directory_rejects_protected_paths_and_nonempty_previous_runs(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            folder = Path(directory)
            with patch.object(runner, "RESEARCH", folder), patch.object(runner, "RUNS", folder / "runs"):
                for name in ("../application", str(folder), "."):
                    with self.subTest(path=name), self.assertRaises(ValueError):
                        runner.create_run_directory(name)
                run_dir = runner.create_run_directory("experiment")
                (run_dir / "saved.json").write_text("keep this", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "previous outputs are preserved"):
                    runner.create_run_directory("experiment")
                self.assertEqual((run_dir / "saved.json").read_text(), "keep this")

    def test_run_overrides_inherited_app_paths_and_restores_environment_on_error(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            folder = Path(directory)
            protected = folder / "application"
            protected.mkdir()
            (protected / "module.py").write_text("original code", encoding="utf-8")
            (folder / "payload.py").write_text('# Recovered source → Python\nvalue = "café"', encoding="utf-8")
            (folder / "stage.py").write_text(
                "import os, sys\nfrom pathlib import Path\n"
                "Path(os.environ['G1_REPO']).mkdir(parents=True)\n"
                "os.chdir(os.environ['G1_REPO'])\n"
                "_write_notebook_file('module.py', 'payload.py')\n"
                "sys.path.insert(0, 'temporary-research-path')\n"
                "raise ValueError('stop after isolated write')\n", encoding="utf-8")
            previous_cwd, previous_path = Path.cwd(), sys.path[:]
            inherited = {"G1_REPO": str(protected), "G1_WORKDIR": str(protected)}
            stages = [{"name": "test", "file": "stage.py", "description": "test"}]
            with patch.dict(os.environ, inherited), patch.object(runner, "RESEARCH", folder), patch.object(runner, "RUNS", folder / "runs"):
                run_dir = runner.create_run_directory("isolated")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "isolated write"):
                        runner.run_research(stages, run_dir, display=print)
                self.assertEqual(os.environ["G1_REPO"], str(protected))
                self.assertEqual(os.environ["G1_WORKDIR"], str(protected))
            self.assertEqual(Path.cwd(), previous_cwd)
            self.assertEqual(sys.path, previous_path)
            self.assertEqual((protected / "module.py").read_text(), "original code")
            recovered = (run_dir / "repository/module.py").read_text(encoding="utf-8")
            self.assertEqual(recovered, '# Recovered source → Python\nvalue = "café"\n')


if __name__ == "__main__":
    unittest.main()
