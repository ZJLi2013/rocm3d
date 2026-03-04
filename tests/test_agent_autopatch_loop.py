from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docker_agent.agent import BuildRequest, DockerAgent  # noqa: E402
from docker_agent.container_runner import RunResult  # noqa: E402


class _FakeRepoManager:
    def clone(self, repo_url: str, repo_dir: Path, depth: int = 1) -> None:  # type: ignore[override]
        repo_dir.mkdir(parents=True, exist_ok=True)


class AgentAutoPatchLoopTest(unittest.TestCase):
    def _make_script(self, td: str) -> str:
        p = Path(td) / "install.sh"
        p.write_text("#!/usr/bin/env bash\necho install\n", encoding="utf-8")
        return str(p)

    @patch("docker_agent.agent.apply_patches_from_plan")
    @patch("docker_agent.agent.analyze_failure_with_llm")
    @patch("docker_agent.agent.run_install_container")
    def test_autopatch_patch_script_then_success(self, mock_run_install, mock_analyze, mock_apply_patch) -> None:
        mock_run_install.side_effect = [
            RunResult(exit_code=1, stdout="err1", stderr="missing ninja", timed_out=False, saved_image=None),
            RunResult(exit_code=0, stdout="ok", stderr="", timed_out=False, saved_image="autorun/fake:latest"),
        ]
        mock_analyze.return_value = {
            "root_cause": {"evidence": ["missing ninja"], "why": "index resolution issue"},
            "execution_plan": {"action": "patch_script", "patches": [{"op": "replace_line", "target": "install_script", "match": "a", "content": "b"}]},
        }
        mock_apply_patch.return_value = {
            "action": "patch_script",
            "changed_files": ["run_injected.sh"],
            "applied": [{"status": "applied"}],
            "errors": [],
        }

        with tempfile.TemporaryDirectory() as td:
            agent = DockerAgent(repo_manager=_FakeRepoManager())
            result = agent.build_image(
                BuildRequest(repo_url="https://github.com/example/repo", base_image="rocm/pytorch:test"),
                script_from_host=self._make_script(td),
                auto_patch_on_fail=True,
                max_auto_patch_retries=3,
                enable_llm_analysis=True,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(mock_run_install.call_count, 2)
        mock_apply_patch.assert_called_once()
        self.assertIsNotNone(result.patch_apply_result)

    @patch("docker_agent.agent.apply_patches_from_plan")
    @patch("docker_agent.agent.analyze_failure_with_llm")
    @patch("docker_agent.agent.run_install_container")
    def test_autopatch_retry_then_success(self, mock_run_install, mock_analyze, mock_apply_patch) -> None:
        mock_run_install.side_effect = [
            RunResult(exit_code=2, stdout="err1", stderr="transient", timed_out=False, saved_image=None),
            RunResult(exit_code=0, stdout="ok", stderr="", timed_out=False, saved_image="autorun/fake:latest"),
        ]
        mock_analyze.return_value = {
            "root_cause": {"evidence": ["transient"], "why": "transient infra issue"},
            "execution_plan": {"action": "retry", "patches": []},
        }

        with tempfile.TemporaryDirectory() as td:
            agent = DockerAgent(repo_manager=_FakeRepoManager())
            result = agent.build_image(
                BuildRequest(repo_url="https://github.com/example/repo", base_image="rocm/pytorch:test"),
                script_from_host=self._make_script(td),
                auto_patch_on_fail=True,
                max_auto_patch_retries=3,
                enable_llm_analysis=True,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(mock_run_install.call_count, 2)
        mock_apply_patch.assert_not_called()
        self.assertIsNone(result.patch_apply_result)

    @patch("docker_agent.agent.apply_patches_from_plan")
    @patch("docker_agent.agent.analyze_failure_with_llm")
    @patch("docker_agent.agent.run_install_container")
    def test_autopatch_need_human_stops_and_returns_need_human(self, mock_run_install, mock_analyze, mock_apply_patch) -> None:
        mock_run_install.return_value = RunResult(
            exit_code=3,
            stdout="err",
            stderr="manual intervention required",
            timed_out=False,
            saved_image=None,
        )
        mock_analyze.return_value = {
            "root_cause": {"evidence": ["manual intervention required"], "why": "cannot auto-fix safely"},
            "execution_plan": {"action": "need_human", "patches": []},
        }

        with tempfile.TemporaryDirectory() as td:
            agent = DockerAgent(repo_manager=_FakeRepoManager())
            result = agent.build_image(
                BuildRequest(repo_url="https://github.com/example/repo", base_image="rocm/pytorch:test"),
                script_from_host=self._make_script(td),
                auto_patch_on_fail=True,
                max_auto_patch_retries=3,
                enable_llm_analysis=True,
            )

        self.assertEqual(result.status, "need_human")
        self.assertEqual(mock_run_install.call_count, 1)
        mock_apply_patch.assert_not_called()
        self.assertEqual((result.llm_analysis_plan or {}).get("execution_plan", {}).get("action"), "need_human")


if __name__ == "__main__":
    unittest.main()
