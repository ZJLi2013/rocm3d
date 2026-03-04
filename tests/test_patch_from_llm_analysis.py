from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from docker_agent.patch_from_llm_analysis import apply_patches_from_llm_analysis


class PatchFromLLMAnalysisTest(unittest.TestCase):
    def _write(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def test_patch_script_replace_line_applies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            install = root / "install.sh"
            analysis = root / "analysis.json"
            self._write(
                install,
                "set -e\npip install gsplat --index-url=https://pypi.amd.com/simple\necho done\n",
            )
            analysis.write_text(
                json.dumps(
                    {
                        "llm_analysis_plan": {
                            "execution_plan": {
                                "action": "patch_script",
                                "patches": [
                                    {
                                        "op": "replace_line",
                                        "target": "install_script",
                                        "match": "pip install gsplat --index-url=https://pypi.amd.com/simple",
                                        "content": "pip install gsplat --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple",
                                    }
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = apply_patches_from_llm_analysis(
                analysis_json_path=str(analysis),
                install_script_path=str(install),
            )
            self.assertEqual(result["action"], "patch_script")
            self.assertEqual(result["errors"], [])
            self.assertEqual(len(result["changed_files"]), 1)
            self.assertIn("--extra-index-url https://pypi.org/simple", install.read_text(encoding="utf-8"))

    def test_non_patch_action_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            install = root / "install.sh"
            analysis = root / "analysis.json"
            self._write(install, "echo install\n")
            analysis.write_text(
                json.dumps(
                    {
                        "llm_analysis_plan": {
                            "execution_plan": {
                                "action": "retry",
                                "patches": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = apply_patches_from_llm_analysis(
                analysis_json_path=str(analysis),
                install_script_path=str(install),
            )
            self.assertEqual(result["action"], "retry")
            self.assertEqual(result["changed_files"], [])
            self.assertEqual(result["errors"], [])

    def test_replace_line_missing_match_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            install = root / "install.sh"
            analysis = root / "analysis.json"
            self._write(install, "echo install\n")
            analysis.write_text(
                json.dumps(
                    {
                        "llm_analysis_plan": {
                            "execution_plan": {
                                "action": "patch_script",
                                "patches": [
                                    {
                                        "op": "replace_line",
                                        "target": "install_script",
                                        "match": "pip install missing",
                                        "content": "pip install fixed",
                                    }
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = apply_patches_from_llm_analysis(
                analysis_json_path=str(analysis),
                install_script_path=str(install),
            )
            self.assertEqual(result["action"], "patch_script")
            self.assertEqual(result["changed_files"], [])
            self.assertTrue(result["errors"])
            self.assertIn("match not found", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
