"""
Slim docker_agent: clone repo only (Path B).

No Dockerfile discovery/synthesis, no docker build, no build_log_summary.
Uses run_install_container / run_sample_container for two-phase flow.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from .repo_manager import RepoManager
from .workspace import Workspace, WorkspaceManager
from .container_runner import run_install_container, run_sample_container
from .error_classifier import classify_failure
from .llm_log_analyzer import analyze_failure_with_llm
from .llm_analyzer import LLMAnalyzer
from .orchestrator import InstallRunOrchestrator
from .patch_from_llm_analysis import apply_patches_from_plan
from .patch_engine import PatchEngine


def _repo_url_to_slug(repo_url: str) -> str:
    """Derive image-safe slug from repo URL, e.g. github.com/naver/dust3r -> naver-dust3r."""
    parsed = urlparse(repo_url)
    path = (parsed.path or "").strip().rstrip("/")
    if path.lower().endswith(".git"):
        path = path[:-4]
    path = path.lstrip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        slug = "-".join(parts[-2:])  # org-repo
    elif len(parts) == 1:
        slug = parts[0]
    else:
        slug = "repo"
    return re.sub(r"[^a-zA-Z0-9._-]", "-", slug).lower()


@dataclass(frozen=True)
class BuildRequest:
    """Request to prepare a repo for run (Path B: base image + in-container install)."""
    repo_url: str
    base_image: Optional[str] = None  # hint for future run step (e.g. rocm/pytorch:...)


@dataclass(frozen=True)
class BuildResult:
    """
    Result of prepare_repo (clone). Optionally includes run result when run was requested.

    - status "success": clone succeeded and (if run was requested) container run exited 0
      and did not time out. status "failed" means clone failed, or run failed / timed out / threw.
    - install_*: first container (install phase) output; commit happens after install on success.
    - run_*: second container (sample run phase) output when run_script_from_host or run_cmd is used.
    """
    status: Literal["success", "failed", "need_human"]
    repo_path: Optional[str] = None
    workspace_path: Optional[str] = None
    message: Optional[str] = None
    install_exit_code: Optional[int] = None
    install_stdout: Optional[str] = None
    install_stderr: Optional[str] = None
    run_exit_code: Optional[int] = None
    run_stdout: Optional[str] = None
    run_stderr: Optional[str] = None
    run_timed_out: Optional[bool] = None
    saved_image: Optional[str] = None  # e.g. autorun/naver-dust3r:latest when commit succeeded
    failure_type: Optional[str] = None
    retryable: Optional[bool] = None
    classifier_confidence: Optional[float] = None
    signals: Optional[list[str]] = None
    llm_analysis_plan: Optional[dict[str, Any]] = None
    patch_apply_result: Optional[dict[str, Any]] = None


class DockerAgent:
    def __init__(
        self,
        *,
        workspace_manager: Optional[WorkspaceManager] = None,
        repo_manager: Optional[RepoManager] = None,
    ) -> None:
        self._workspace_manager = workspace_manager or WorkspaceManager()
        self._repo_manager = repo_manager or RepoManager()

    def build_image(
        self,
        req: BuildRequest,
        *,
        keep_workspace: bool = True,
        script_from_host: Optional[str] = None,
        run_script_from_host: Optional[str] = None,
        run_cmd: Optional[str] = None,
        install_cmd: Optional[str] = None,
        run_timeout_sec: int = 3600,
        commit_image: bool = True,
        enable_llm_analysis: bool = False,
        llm_timeout_sec: float = 90.0,
        auto_patch_on_fail: bool = False,
        max_auto_patch_retries: int = 3,
    ) -> BuildResult:
        """
        Two-phase when run_script_from_host or run_cmd is set: (1) install script -> commit;
        (2) run sample in saved image. Otherwise: single install run -> commit.
        script_from_host: install script on host; copied to repo as run_injected.sh.
        run_script_from_host: sample run script on host; run in second container from saved image.
        """
        def _progress(msg: str) -> None:
            print(f"[docker_agent] {msg}", file=sys.stderr, flush=True)

        start_ts = time.monotonic()
        _progress(f"prepare_repo: repo_url={req.repo_url}")

        ws = self._workspace_manager.create(keep=keep_workspace)
        assert isinstance(ws, Workspace)
        repo_dir = ws.path / "repo"

        try:
            _progress(f"cloning repo -> {repo_dir}")
            self._repo_manager.clone(req.repo_url, repo_dir, depth=1)
            _progress("clone complete")

            repo_path_str = str(repo_dir.resolve())
            ws_path_str = str(ws.path.resolve())

            script_to_run: Optional[str] = None
            if script_from_host:
                host_script = Path(script_from_host)
                if not host_script.is_file():
                    raise FileNotFoundError(f"script_from_host not found: {script_from_host}")
                injected = repo_dir / "run_injected.sh"
                shutil.copy2(host_script, injected)
                injected.chmod(0o755)
                script_to_run = "run_injected.sh"

            if req.base_image and (script_to_run or install_cmd or run_cmd):
                _progress(f"install phase: image={req.base_image}")
                commit_image_as: Optional[str] = None
                if commit_image:
                    slug = _repo_url_to_slug(req.repo_url)
                    commit_image_as = f"autorun/{slug}"
                try:
                    run_script_host: Optional[str] = None
                    if run_script_from_host:
                        run_script_host = str(Path(run_script_from_host).resolve())
                    orchestrator = InstallRunOrchestrator(
                        run_install=run_install_container,
                        run_sample=run_sample_container,
                        classify_failure=classify_failure,
                        analyzer=LLMAnalyzer(analyze_fn=analyze_failure_with_llm),
                        patch_engine=PatchEngine(apply_fn=apply_patches_from_plan),
                        progress=_progress,
                    )
                    outcome = orchestrator.execute(
                        repo_dir=repo_dir,
                        repo_url=req.repo_url,
                        base_image=req.base_image,
                        script_to_run=script_to_run,
                        script_from_host=script_from_host,
                        run_script_host=run_script_host,
                        run_cmd=run_cmd,
                        install_cmd=install_cmd,
                        run_timeout_sec=run_timeout_sec,
                        commit_image_as=commit_image_as,
                        enable_llm_analysis=enable_llm_analysis,
                        llm_timeout_sec=llm_timeout_sec,
                        auto_patch_on_fail=auto_patch_on_fail,
                        max_auto_patch_retries=max_auto_patch_retries,
                    )
                    sample_result = outcome.sample_result
                    classification = outcome.classification
                    return BuildResult(
                        status=outcome.status,
                        repo_path=repo_path_str,
                        workspace_path=ws_path_str,
                        message=outcome.message,
                        install_exit_code=outcome.install_result.exit_code,
                        install_stdout=outcome.install_result.stdout,
                        install_stderr=outcome.install_result.stderr,
                        run_exit_code=sample_result.exit_code if sample_result else None,
                        run_stdout=sample_result.stdout if sample_result else None,
                        run_stderr=sample_result.stderr if sample_result else None,
                        run_timed_out=sample_result.timed_out if sample_result else None,
                        saved_image=outcome.install_result.saved_image,
                        failure_type=classification.failure_type if classification else None,
                        retryable=classification.retryable if classification else None,
                        classifier_confidence=classification.confidence if classification else None,
                        signals=[s.matched_line for s in classification.signals] if classification else None,
                        llm_analysis_plan=outcome.llm_analysis_plan,
                        patch_apply_result=outcome.patch_apply_result,
                    )
                except Exception as e:
                    _progress(f"install/sample container failed: {e}")
                    return BuildResult(
                        status="failed",
                        repo_path=repo_path_str,
                        workspace_path=ws_path_str,
                        message=f"install/sample container failed: {e}",
                        install_exit_code=-1,
                        install_stdout=None,
                        install_stderr=None,
                        run_exit_code=None,
                        run_stdout=None,
                        run_stderr=None,
                        run_timed_out=None,
                        saved_image=None,
                    )

            elapsed = time.monotonic() - start_ts
            _progress(f"done (elapsed={elapsed:.2f}s)")
            return BuildResult(
                status="success",
                repo_path=repo_path_str,
                workspace_path=ws_path_str,
                message=None,
            )
        except Exception as e:
            _progress(f"prepare_repo failed: {e}")
            return BuildResult(
                status="failed",
                repo_path=None,
                workspace_path=str(ws.path.resolve()),
                message=str(e),
            )
        finally:
            if not keep_workspace:
                ws.cleanup()
