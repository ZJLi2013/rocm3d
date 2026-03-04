from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .error_classifier import ErrorClassification
from .llm_log_analyzer import analyze_failure_with_llm


@dataclass(frozen=True)
class LLMFailureContext:
    stage: str
    repo_url: str
    base_image: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]
    script_text: Optional[str]


class LLMAnalyzer:
    """Thin analyzer layer that isolates LLM planning from orchestration."""

    def __init__(
        self,
        *,
        analyze_fn: Callable[..., dict] = analyze_failure_with_llm,
    ) -> None:
        self._analyze_fn = analyze_fn

    def analyze(
        self,
        *,
        context: LLMFailureContext,
        classification: ErrorClassification,
        timeout_sec: float,
    ) -> dict:
        return self._analyze_fn(
            stage=context.stage,  # type: ignore[arg-type]
            repo_url=context.repo_url,
            base_image=context.base_image,
            classification=classification,
            stdout=context.stdout,
            stderr=context.stderr,
            script_text=context.script_text,
            timeout_sec=timeout_sec,
        )
