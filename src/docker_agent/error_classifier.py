from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


FailureType = Literal[
    "timeout",
    "dependency",
    "script_path",
    "apt_package",
    "network",
    "permission",
    "unknown",
]

SignalSource = Literal["stdout_snippet", "stdout_tail", "stderr_tail"]


@dataclass(frozen=True)
class ErrorSignal:
    pattern: str
    matched_line: str
    source: SignalSource


@dataclass(frozen=True)
class LogExtract:
    stdout_error_snippets: list[str]
    stdout_tail: list[str]
    stderr_tail: list[str]


@dataclass(frozen=True)
class ErrorClassification:
    failure_type: FailureType
    retryable: bool
    confidence: float
    signals: list[ErrorSignal]


_ERROR_SNIPPET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("dependency.no_matching_dist", re.compile(r"no matching distribution found", re.IGNORECASE)),
    ("dependency.cannot_find_version", re.compile(r"could not find a version that satisfies", re.IGNORECASE)),
    ("script_path.cd_missing_dir", re.compile(r"cd: .*no such file or directory", re.IGNORECASE)),
    ("apt.package_not_found", re.compile(r"unable to locate package", re.IGNORECASE)),
    ("network.transient", re.compile(r"temporary failure resolving|connection reset|read timed out|service unavailable", re.IGNORECASE)),
    ("permission.denied", re.compile(r"permission denied|operation not permitted", re.IGNORECASE)),
    ("python.traceback", re.compile(r"traceback \(most recent call last\):", re.IGNORECASE)),
    ("generic.error", re.compile(r"\berror\b", re.IGNORECASE)),
    ("generic.failed", re.compile(r"\bfailed\b", re.IGNORECASE)),
]


_CLASSIFIER_RULES: list[tuple[FailureType, str, re.Pattern[str], bool, float]] = [
    ("dependency", "dependency.no_matching_dist", re.compile(r"no matching distribution found", re.IGNORECASE), True, 0.95),
    ("dependency", "dependency.cannot_find_version", re.compile(r"could not find a version that satisfies", re.IGNORECASE), True, 0.95),
    ("script_path", "script_path.cd_missing_dir", re.compile(r"cd: .*no such file or directory", re.IGNORECASE), True, 0.95),
    ("apt_package", "apt.package_not_found", re.compile(r"unable to locate package", re.IGNORECASE), True, 0.95),
    ("network", "network.transient", re.compile(r"temporary failure resolving|connection reset|read timed out|service unavailable", re.IGNORECASE), True, 0.85),
    ("permission", "permission.denied", re.compile(r"permission denied|operation not permitted", re.IGNORECASE), False, 0.90),
]


def _dedupe_keep_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.strip()
        if not key:
            continue
        # High-noise warning; keep only first instance.
        if "running pip as the 'root' user" in key.lower():
            key = "pip_root_warning"
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def extract_error_logs(
    *,
    stdout: str | None,
    stderr: str | None,
    stdout_tail_lines: int = 120,
    stderr_tail_lines: int = 120,
    stdout_error_snippet_limit: int = 30,
) -> LogExtract:
    stdout_lines = (stdout or "").splitlines()
    stderr_lines = (stderr or "").splitlines()

    stdout_tail = stdout_lines[-stdout_tail_lines:] if stdout_tail_lines > 0 else []
    stderr_tail = stderr_lines[-stderr_tail_lines:] if stderr_tail_lines > 0 else []

    snippet_hits: list[str] = []
    for line in stdout_lines:
        for _, pattern in _ERROR_SNIPPET_PATTERNS:
            if pattern.search(line):
                snippet_hits.append(line)
                break
    snippet_hits = _dedupe_keep_order(snippet_hits)
    if stdout_error_snippet_limit > 0:
        snippet_hits = snippet_hits[:stdout_error_snippet_limit]

    return LogExtract(
        stdout_error_snippets=snippet_hits,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def classify_failure(
    *,
    exit_code: int | None,
    timed_out: bool | None,
    stdout: str | None,
    stderr: str | None,
) -> ErrorClassification:
    if bool(timed_out):
        return ErrorClassification(
            failure_type="timeout",
            retryable=True,
            confidence=1.0,
            signals=[ErrorSignal(pattern="timeout", matched_line=f"timed_out=True exit_code={exit_code}", source="stderr_tail")],
        )

    extracted = extract_error_logs(stdout=stdout, stderr=stderr)
    ordered_lines: list[tuple[SignalSource, str]] = []
    ordered_lines.extend(("stderr_tail", line) for line in extracted.stderr_tail)
    ordered_lines.extend(("stdout_snippet", line) for line in extracted.stdout_error_snippets)
    ordered_lines.extend(("stdout_tail", line) for line in extracted.stdout_tail)

    for failure_type, pattern_name, rule, retryable, confidence in _CLASSIFIER_RULES:
        for source, line in ordered_lines:
            if rule.search(line):
                return ErrorClassification(
                    failure_type=failure_type,
                    retryable=retryable,
                    confidence=confidence,
                    signals=[ErrorSignal(pattern=pattern_name, matched_line=line, source=source)],
                )

    fallback_signals: list[ErrorSignal] = []
    for source, line in ordered_lines[:3]:
        fallback_signals.append(ErrorSignal(pattern="unknown", matched_line=line, source=source))

    return ErrorClassification(
        failure_type="unknown",
        retryable=False,
        confidence=0.2,
        signals=fallback_signals,
    )
