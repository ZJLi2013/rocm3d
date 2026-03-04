from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docker_agent.error_classifier import classify_failure, extract_error_logs  # noqa: E402


class ErrorClassifierFromSamplesTest(unittest.TestCase):
    def _load_sample(self, name: str) -> dict:
        p = ROOT / "samples" / "test_output" / name
        return json.loads(p.read_text(encoding="utf-8"))

    def test_dependency_failure_from_depth_anything_3(self) -> None:
        data = self._load_sample("depth_anything_3.json")
        result = classify_failure(
            exit_code=data.get("install_exit_code"),
            timed_out=False,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
        )
        self.assertEqual(result.failure_type, "dependency")
        self.assertTrue(result.retryable)
        self.assertTrue(any("No matching distribution found" in s.matched_line for s in result.signals))

    def test_script_path_failure_from_fast3r(self) -> None:
        data = self._load_sample("fast3r.json")
        result = classify_failure(
            exit_code=data.get("install_exit_code"),
            timed_out=False,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
        )
        self.assertEqual(result.failure_type, "script_path")
        self.assertTrue(result.retryable)
        self.assertTrue(any("No such file or directory" in s.matched_line for s in result.signals))

    def test_apt_package_failure_from_partcrafter(self) -> None:
        data = self._load_sample("partcrafter.json")
        result = classify_failure(
            exit_code=data.get("install_exit_code"),
            timed_out=False,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
        )
        self.assertEqual(result.failure_type, "apt_package")
        self.assertTrue(result.retryable)
        self.assertTrue(any("Unable to locate package" in s.matched_line for s in result.signals))

    def test_stdout_extraction_does_not_depend_on_full_log(self) -> None:
        data = self._load_sample("ml_sharp.json")
        extract = extract_error_logs(
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
            stdout_tail_lines=50,
            stderr_tail_lines=50,
            stdout_error_snippet_limit=10,
        )
        self.assertLessEqual(len(extract.stdout_tail), 50)
        self.assertLessEqual(len(extract.stdout_error_snippets), 10)
        self.assertTrue(
            any("No matching distribution found" in line for line in extract.stdout_error_snippets),
            "Expected dependency error hint in extracted snippets",
        )


if __name__ == "__main__":
    unittest.main()
