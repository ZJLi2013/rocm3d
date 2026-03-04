from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from .container_runner import RunResult
from .error_classifier import ErrorClassification
from .llm_analyzer import LLMAnalyzer, LLMFailureContext
from .patch_engine import PatchApplyRequest, PatchEngine


@dataclass(frozen=True)
class OrchestrationOutcome:
    status: Literal["success", "failed", "need_human"]
    message: Optional[str]
    install_result: RunResult
    sample_result: Optional[RunResult]
    classification: Optional[ErrorClassification]
    llm_analysis_plan: Optional[dict[str, Any]]
    patch_apply_result: Optional[dict[str, Any]]


class InstallRunOrchestrator:
    """Orchestrates install/run flow with optional LLM analyze + patch loop."""

    def __init__(
        self,
        *,
        run_install: Callable[..., RunResult],
        run_sample: Callable[..., RunResult],
        classify_failure: Callable[..., ErrorClassification],
        analyzer: LLMAnalyzer,
        patch_engine: PatchEngine,
        progress: Callable[[str], None],
    ) -> None:
        self._run_install = run_install
        self._run_sample = run_sample
        self._classify_failure = classify_failure
        self._analyzer = analyzer
        self._patch_engine = patch_engine
        self._progress = progress

    def _read_text_if_exists(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8", errors="replace")

    def execute(
        self,
        *,
        repo_dir: Path,
        repo_url: str,
        base_image: str,
        script_to_run: Optional[str],
        script_from_host: Optional[str],
        run_script_host: Optional[str],
        run_cmd: Optional[str],
        install_cmd: Optional[str],
        run_timeout_sec: int,
        commit_image_as: Optional[str],
        enable_llm_analysis: bool,
        llm_timeout_sec: float,
        auto_patch_on_fail: bool,
        max_auto_patch_retries: int,
    ) -> OrchestrationOutcome:
        two_phase = bool(run_script_host or (run_cmd and script_to_run))
        first_run_cmd = run_cmd if not two_phase else None

        max_attempts = 1 + max(0, max_auto_patch_retries) if auto_patch_on_fail else 1
        install_result: Optional[RunResult] = None
        install_ok = False
        classification: Optional[ErrorClassification] = None
        llm_plan: Optional[dict[str, Any]] = None
        patch_apply_result: Optional[dict[str, Any]] = None

        for attempt in range(1, max_attempts + 1):
            self._progress(f"install attempt {attempt}/{max_attempts}")
            install_result = self._run_install(
                base_image,
                str(repo_dir.resolve()),
                script_path=script_to_run,
                install_cmd=install_cmd,
                run_cmd=first_run_cmd,
                timeout_sec=run_timeout_sec,
                commit_image_as=commit_image_as,
            )
            self._progress(f"install exit_code={install_result.exit_code} timed_out={install_result.timed_out}")
            install_ok = install_result.exit_code == 0 and not install_result.timed_out
            if install_result.saved_image:
                self._progress(f"saved_image={install_result.saved_image}")
            if install_ok:
                break

            classification = self._classify_failure(
                exit_code=install_result.exit_code,
                timed_out=install_result.timed_out,
                stdout=install_result.stdout,
                stderr=install_result.stderr,
            )
            llm_plan = None
            if enable_llm_analysis:
                install_script_text = self._read_text_if_exists(
                    str((repo_dir / script_to_run).resolve()) if script_to_run else script_from_host
                )
                llm_plan = self._analyzer.analyze(
                    context=LLMFailureContext(
                        stage="install",
                        repo_url=repo_url,
                        base_image=base_image,
                        stdout=install_result.stdout,
                        stderr=install_result.stderr,
                        script_text=install_script_text,
                    ),
                    classification=classification,
                    timeout_sec=llm_timeout_sec,
                )

            if not auto_patch_on_fail or attempt >= max_attempts:
                break
            if llm_plan is None:
                self._progress("auto-patch enabled but llm analysis is unavailable; stopping retries")
                break

            action = str((llm_plan.get("execution_plan") or {}).get("action", "need_human"))
            if action == "patch_script":
                if not script_to_run:
                    self._progress("patch_script requested but install script path is unavailable; stopping retries")
                    break
                patch_apply_result = self._patch_engine.apply(
                    PatchApplyRequest(
                        llm_plan=llm_plan,
                        install_script_path=str((repo_dir / script_to_run).resolve()),
                    )
                )
                if patch_apply_result.get("errors"):
                    self._progress(f"patch apply failed: {patch_apply_result.get('errors')}")
                    break
                self._progress("patch applied; rerunning install")
                continue
            if action == "retry":
                self._progress("llm suggested retry; rerunning install without patch")
                continue
            self._progress(f"llm action={action}; stopping auto loop")
            break

        assert install_result is not None

        do_sample = install_ok and bool(install_result.saved_image) and two_phase
        if do_sample:
            self._progress(f"sample phase: image={install_result.saved_image}")
            sample_result = self._run_sample(
                install_result.saved_image,
                run_script_host_path=run_script_host,
                run_cmd=run_cmd if not run_script_host else None,
                timeout_sec=run_timeout_sec,
            )
            self._progress(f"sample exit_code={sample_result.exit_code} timed_out={sample_result.timed_out}")
            sample_ok = sample_result.exit_code == 0 and not sample_result.timed_out
            classification = None
            llm_plan_run = None
            if not sample_ok:
                classification = self._classify_failure(
                    exit_code=sample_result.exit_code,
                    timed_out=sample_result.timed_out,
                    stdout=sample_result.stdout,
                    stderr=sample_result.stderr,
                )
            if enable_llm_analysis and classification is not None:
                run_script_text = self._read_text_if_exists(run_script_host)
                llm_plan_run = self._analyzer.analyze(
                    context=LLMFailureContext(
                        stage="run",
                        repo_url=repo_url,
                        base_image=base_image,
                        stdout=sample_result.stdout,
                        stderr=sample_result.stderr,
                        script_text=run_script_text,
                    ),
                    classification=classification,
                    timeout_sec=llm_timeout_sec,
                )
            return OrchestrationOutcome(
                status="success" if sample_ok else "failed",
                message=None if sample_ok else ("sample timed out" if sample_result.timed_out else "sample failed"),
                install_result=install_result,
                sample_result=sample_result,
                classification=classification,
                llm_analysis_plan=llm_plan_run,
                patch_apply_result=patch_apply_result,
            )

        status_value: Literal["success", "failed", "need_human"] = "success" if install_ok else "failed"
        if auto_patch_on_fail and not install_ok:
            status_value = "need_human"
        return OrchestrationOutcome(
            status=status_value,
            message=(
                None
                if install_ok
                else (
                    "install failed after auto patch retries"
                    if auto_patch_on_fail
                    else ("install timed out" if install_result.timed_out else "install failed")
                )
            ),
            install_result=install_result,
            sample_result=None,
            classification=classification,
            llm_analysis_plan=llm_plan,
            patch_apply_result=patch_apply_result,
        )
