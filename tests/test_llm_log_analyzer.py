from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docker_agent.error_classifier import classify_failure  # noqa: E402
from docker_agent.llm_log_analyzer import analyze_failure_with_llm, parse_llm_analysis  # noqa: E402


class LLMLogAnalyzerParseTest(unittest.TestCase):
    def _load_json(self, name: str) -> dict:
        p = ROOT / "samples" / "test_output" / name
        return json.loads(p.read_text(encoding="utf-8"))

    def test_parse_plain_json(self) -> None:
        raw = """
{
  "root_cause": {
    "evidence": ["No matching distribution found for ninja"],
    "why": "pip index cannot resolve transitive dependency ninja"
  },
  "execution_plan": {
    "action": "patch_script",
    "patches": [
      {
        "op": "append_block",
        "target": "install_script",
        "match": "",
        "content": "python -m pip install ninja"
      }
    ]
  }
}
"""
        plan = parse_llm_analysis(raw)
        self.assertEqual(plan.execution_plan.action, "patch_script")
        self.assertEqual(len(plan.execution_plan.patches), 1)
        self.assertIn("ninja", plan.root_cause.why)

    def test_parse_json_fenced_block(self) -> None:
        raw = """some text
```json
{"root_cause":{"evidence":["timeout"],"why":"environment unstable"},"execution_plan":{"action":"need_human","patches":[]}}
```
"""
        plan = parse_llm_analysis(raw)
        self.assertEqual(plan.execution_plan.action, "need_human")
        self.assertEqual(plan.root_cause.why, "environment unstable")

    def test_parse_invalid_json_fallback(self) -> None:
        raw = "not-json-response"
        plan = parse_llm_analysis(raw)
        self.assertEqual(plan.execution_plan.action, "need_human")
        self.assertEqual(plan.root_cause.why, "LLM response is not valid JSON")

    @patch("docker_agent.llm_log_analyzer.get_default_client")
    def test_depth_anything_case_to_llm_plan(self, mock_get_client) -> None:
        class _FakeClient:
            def chat(self, **kwargs):  # type: ignore[no-untyped-def]
                return json.dumps(
                    {
                        "root_cause": {
                            "evidence": ["No matching distribution found for ninja"],
                            "why": "Index configuration misses ninja package",
                        },
                        "execution_plan": {
                            "action": "patch_script",
                            "patches": [
                                {
                                    "op": "append_block",
                                    "target": "install_script",
                                    "match": "",
                                    "content": "pip install ninja -i https://pypi.org/simple",
                                }
                            ],
                        },
                    }
                )

        mock_get_client.return_value = _FakeClient()
        data = self._load_json("depth_anything_3.json")
        classification = classify_failure(
            exit_code=data.get("install_exit_code"),
            timed_out=False,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
        )
        plan = analyze_failure_with_llm(
            stage="install",
            repo_url="https://github.com/ByteDance-Seed/Depth-Anything-3",
            base_image="rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0",
            classification=classification,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
            script_text=None,
            timeout_sec=5.0,
        )
        self.assertEqual(classification.failure_type, "dependency")
        self.assertEqual(plan["execution_plan"]["action"], "patch_script")

    @patch("docker_agent.llm_log_analyzer.get_default_client")
    def test_fast3r_case_to_llm_plan(self, mock_get_client) -> None:
        class _FakeClient:
            def chat(self, **kwargs):  # type: ignore[no-untyped-def]
                return json.dumps(
                    {
                        "root_cause": {
                            "evidence": ["No such file or directory"],
                            "why": "Script path assumes wrong working directory",
                        },
                        "execution_plan": {
                            "action": "patch_script",
                            "patches": [
                                {
                                    "op": "replace_line",
                                    "target": "install_script",
                                    "match": "cd croco",
                                    "content": "cd src/croco",
                                }
                            ],
                        },
                    }
                )

        mock_get_client.return_value = _FakeClient()
        data = self._load_json("fast3r_rerun.json")
        classification = classify_failure(
            exit_code=data.get("install_exit_code"),
            timed_out=False,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
        )
        plan = analyze_failure_with_llm(
            stage="install",
            repo_url="https://github.com/facebookresearch/fast3r",
            base_image="rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0",
            classification=classification,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
            script_text=None,
            timeout_sec=5.0,
        )
        self.assertEqual(classification.failure_type, "script_path")
        self.assertEqual(plan["execution_plan"]["action"], "patch_script")

    @patch("docker_agent.llm_log_analyzer.get_default_client")
    def test_llm_call_retries_and_then_succeeds(self, mock_get_client) -> None:
        from llms import AMDGatewayClientError

        class _RetryClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, **kwargs):  # type: ignore[no-untyped-def]
                self.calls += 1
                if self.calls < 3:
                    raise AMDGatewayClientError("temporary network issue")
                return json.dumps(
                    {
                        "root_cause": {
                            "evidence": ["temporary network issue"],
                            "why": "Transient transport issue",
                        },
                        "execution_plan": {
                            "action": "retry",
                            "patches": [],
                        },
                    }
                )

        client = _RetryClient()
        mock_get_client.return_value = client
        data = self._load_json("depth_anything_3.json")
        classification = classify_failure(
            exit_code=data.get("install_exit_code"),
            timed_out=False,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
        )
        plan = analyze_failure_with_llm(
            stage="install",
            repo_url="https://github.com/ByteDance-Seed/Depth-Anything-3",
            base_image="rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0",
            classification=classification,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
            script_text=None,
            timeout_sec=5.0,
            max_retries=3,
        )
        self.assertEqual(plan["execution_plan"]["action"], "retry")
        self.assertEqual(client.calls, 3)

    @patch("docker_agent.llm_log_analyzer.get_default_client")
    def test_llm_call_retries_gateway_unreachable_suggests_switch_node(self, mock_get_client) -> None:
        from llms import AMDGatewayClientError

        class _AlwaysResetClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, **kwargs):  # type: ignore[no-untyped-def]
                self.calls += 1
                raise AMDGatewayClientError(
                    "Error calling AMD gateway: <urlopen error [Errno 104] Connection reset by peer>"
                )

        client = _AlwaysResetClient()
        mock_get_client.return_value = client
        data = self._load_json("depth_anything_3.json")
        classification = classify_failure(
            exit_code=data.get("install_exit_code"),
            timed_out=False,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
        )
        plan = analyze_failure_with_llm(
            stage="install",
            repo_url="https://github.com/ByteDance-Seed/Depth-Anything-3",
            base_image="rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0",
            classification=classification,
            stdout=data.get("install_stdout"),
            stderr=data.get("install_stderr"),
            script_text=None,
            timeout_sec=5.0,
            max_retries=3,
        )
        self.assertEqual(plan["execution_plan"]["action"], "need_human")
        self.assertIn("stderr/stdout", plan["root_cause"]["why"])
        self.assertIn("switch to another node", plan["root_cause"]["why"])
        self.assertEqual(client.calls, 3)

    @patch("docker_agent.llm_log_analyzer.get_default_client")
    def test_prompt_payload_includes_experience_snippets(self, mock_get_client) -> None:
        captured_user_payload: dict | None = None

        class _CaptureClient:
            def chat(self, **kwargs):  # type: ignore[no-untyped-def]
                nonlocal captured_user_payload
                user_msg = kwargs["messages"][1]["content"]
                captured_user_payload = json.loads(user_msg)
                return json.dumps(
                    {
                        "root_cause": {"evidence": ["x"], "why": "y"},
                        "execution_plan": {"action": "retry", "patches": []},
                    }
                )

        mock_get_client.return_value = _CaptureClient()
        classification = classify_failure(
            exit_code=1,
            timed_out=False,
            stdout="E: Unable to locate package libegl1-mesa",
            stderr="",
        )

        with tempfile.TemporaryDirectory() as td:
            exp_file = Path(td) / "dependency.md"
            exp_file.write_text(
                "- Prefer --extra-index-url for transitive dependencies\n"
                "- Verify package name changes across distro versions\n",
                encoding="utf-8",
            )
            with patch("docker_agent.llm_log_analyzer._EXPERIENCE_DIR", Path(td)):
                analyze_failure_with_llm(
                    stage="install",
                    repo_url="https://github.com/ByteDance-Seed/Depth-Anything-3",
                    base_image="rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0",
                    classification=classification,
                    stdout="E: Unable to locate package libegl1-mesa",
                    stderr="",
                    script_text=None,
                    timeout_sec=5.0,
                )

        self.assertIsNotNone(captured_user_payload)
        assert captured_user_payload is not None
        self.assertIn("experience_snippets", captured_user_payload)
        self.assertGreaterEqual(len(captured_user_payload["experience_snippets"]), 1)


if __name__ == "__main__":
    unittest.main()
