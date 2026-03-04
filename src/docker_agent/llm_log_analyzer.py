from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from llms import LLMError as AMDGatewayClientError, get_default_client

from .error_classifier import ErrorClassification, extract_error_logs


ExecutionAction = Literal["patch_script", "retry", "need_human"]
PatchOp = Literal["replace_line", "append_block", "prepend_block"]
PatchTarget = Literal["install_script", "run_script"]

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_SYSTEM_PROMPT = (
    "You are a software installation failure analyst. "
    "Return only valid JSON that matches the requested schema."
)
_DEFAULT_CONSTRAINTS = [
    "Return JSON only.",
    "Do not modify repository source code; only install/run script plan.",
    "Avoid destructive commands.",
    "Do not suppress or downgrade errors in scripts (forbidden patterns include: set +e, '|| true', '|| echo ...', 'exit 0' after failures).",
    "For dependency installation failures, prefer actionable fixes (e.g., correct package names, apt index refresh, distro-specific alternatives). If not confidently fixable, choose need_human.",
]
_DEFAULT_OUTPUT_SCHEMA: dict[str, Any] = {
    "root_cause": {
        "evidence": ["string", "..."],
        "why": "string",
    },
    "execution_plan": {
        "action": "patch_script | retry | need_human",
        "patches": [
            {
                "op": "replace_line | append_block | prepend_block",
                "target": "install_script | run_script",
                "match": "string",
                "content": "string",
            }
        ],
    },
}


def _read_prompt_text(filename: str, *, default: str) -> str:
    path = _PROMPTS_DIR / filename
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or default
    except Exception:
        return default


def _read_prompt_constraints() -> list[str]:
    raw = _read_prompt_text("analyzer_constraints.txt", default="\n".join(_DEFAULT_CONSTRAINTS))
    constraints = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
    return constraints or list(_DEFAULT_CONSTRAINTS)


def _read_output_schema() -> dict[str, Any]:
    path = _PROMPTS_DIR / "analyzer_output_schema.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass
    return _DEFAULT_OUTPUT_SCHEMA


def _read_fewshot_text() -> str:
    return _read_prompt_text("analyzer_fewshot.md", default="")




@dataclass(frozen=True)
class LLMRootCause:
    evidence: list[str]
    why: str


@dataclass(frozen=True)
class LLMScriptPatch:
    op: PatchOp
    target: PatchTarget
    match: str
    content: str


@dataclass(frozen=True)
class LLMExecutionPlan:
    action: ExecutionAction
    patches: list[LLMScriptPatch]


@dataclass(frozen=True)
class LLMAnalysisPlan:
    root_cause: LLMRootCause
    execution_plan: LLMExecutionPlan


def _clip_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _validate_replace_line_matches(plan: LLMAnalysisPlan, script_text: Optional[str]) -> list[str]:
    """Return a list of error strings for every replace_line match not found in script_text.

    Only validates `replace_line` patches; append/prepend anchors are advisory.
    Returns an empty list when all matches are found (or there are no replace_line patches).
    """
    if not script_text:
        return []
    script_lines = {line.rstrip("\r\n") for line in script_text.splitlines()}
    missing: list[str] = []
    for patch in plan.execution_plan.patches:
        if patch.op == "replace_line" and patch.match and patch.match not in script_lines:
            missing.append(patch.match)
    return missing


def _extract_json_block(raw: str) -> str:
    # 支持 ```json ... ``` 与直接 JSON 两种输出
    code_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _looks_like_gateway_connectivity_issue(err_msg: str) -> bool:
    text = err_msg.lower()
    network_signals = [
        "connection reset",
        "connection refused",
        "timed out",
        "timeout",
        "no route to host",
        "name or service not known",
        "temporary failure in name resolution",
        "ssl",
        "tls",
        "urlopen error",
    ]
    return any(sig in text for sig in network_signals)


def _fallback_plan(*, why: str, evidence: Optional[list[str]] = None) -> LLMAnalysisPlan:
    return LLMAnalysisPlan(
        root_cause=LLMRootCause(evidence=evidence or [], why=why),
        execution_plan=LLMExecutionPlan(action="need_human", patches=[]),
    )


def parse_llm_analysis(raw: str) -> LLMAnalysisPlan:
    parsed_text = _extract_json_block(raw)
    try:
        data = json.loads(parsed_text)
    except Exception:
        return _fallback_plan(why="LLM response is not valid JSON", evidence=[raw.strip()[:500]])

    root_obj = data.get("root_cause") or {}
    if not isinstance(root_obj, dict):
        root_obj = {}

    raw_evidence = root_obj.get("evidence", [])
    evidence: list[str] = []
    if isinstance(raw_evidence, list):
        evidence = [str(x) for x in raw_evidence if str(x).strip()]
    why = str(root_obj.get("why", "")).strip()
    if not why:
        why = "No root cause explanation provided by LLM"

    execution_obj = data.get("execution_plan") or {}
    if not isinstance(execution_obj, dict):
        execution_obj = {}

    action = str(execution_obj.get("action", "need_human"))
    if action not in {"patch_script", "retry", "need_human"}:
        action = "need_human"

    patches: list[LLMScriptPatch] = []
    for item in execution_obj.get("patches", []) or []:
        op = str(item.get("op", "append_block"))
        target = str(item.get("target", "install_script"))
        if op not in {"replace_line", "append_block", "prepend_block"}:
            continue
        if target not in {"install_script", "run_script"}:
            continue
        patches.append(
            LLMScriptPatch(
                op=op,  # type: ignore[arg-type]
                target=target,  # type: ignore[arg-type]
                match=str(item.get("match", "")),
                content=str(item.get("content", "")),
            )
        )

    return LLMAnalysisPlan(
        root_cause=LLMRootCause(evidence=evidence, why=why),
        execution_plan=LLMExecutionPlan(
            action=action,  # type: ignore[arg-type]
            patches=patches,
        ),
    )


def analyze_failure_with_llm(
    *,
    stage: Literal["install", "run"],
    repo_url: str,
    base_image: Optional[str],
    classification: ErrorClassification,
    stdout: Optional[str],
    stderr: Optional[str],
    script_text: Optional[str],
    timeout_sec: float = 90.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    extracted = extract_error_logs(stdout=stdout, stderr=stderr)
    fewshot_text = _read_fewshot_text()
    prompt_payload = {
        "task": "analyze_install_or_run_failure_and_plan_fix",
        "stage": stage,
        "repo_url": repo_url,
        "base_image": base_image,
        "classification": {
            "failure_type": classification.failure_type,
            "retryable": classification.retryable,
            "confidence": classification.confidence,
            "signals": [s.matched_line for s in classification.signals],
        },
        "script_text_tail": _clip_lines(script_text or "", 120),
        "stdout_error_snippets": extracted.stdout_error_snippets,
        "stdout_tail": extracted.stdout_tail[-120:],
        "stderr_tail": extracted.stderr_tail[-120:],
        "constraints": _read_prompt_constraints(),
        "output_schema": _read_output_schema(),
        "fewshot_examples": fewshot_text,
    }

    system = _read_prompt_text("analyzer_system.txt", default=_DEFAULT_SYSTEM_PROMPT)
    user = json.dumps(prompt_payload, ensure_ascii=False)

    try:
        client = get_default_client()
    except AMDGatewayClientError as e:
        return asdict(_fallback_plan(why=f"LLM call failed: {e}"))
    except Exception as e:
        return asdict(_fallback_plan(why=f"Unexpected analyzer error: {e}"))

    attempts = max(1, max_retries)
    last_error: Optional[AMDGatewayClientError] = None
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for _ in range(attempts):
        try:
            raw = client.chat(
                messages=messages,
                timeout_sec=timeout_sec,
                temperature=1.0,
            )
            plan = parse_llm_analysis(raw)
            missing_matches = _validate_replace_line_matches(plan, script_text)
            if missing_matches:
                # Feed the validation error back as a follow-up user message so the
                # LLM can correct its patch using only lines that actually exist.
                correction = (
                    "Your previous response contained replace_line patches whose `match` "
                    "strings were NOT found verbatim in script_text_tail. "
                    "These match values are invalid:\n"
                    + "\n".join(f"  - {m!r}" for m in missing_matches)
                    + "\n\nInstructions to fix:\n"
                    "1. Re-read script_text_tail carefully.\n"
                    "2. For replace_line: copy the exact line from the script (including leading spaces).\n"
                    "3. If no exact matching line exists, use append_block or prepend_block with a real anchor line instead.\n"
                    "Return the corrected JSON plan only."
                )
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": correction},
                ]
                continue
            return asdict(plan)
        except AMDGatewayClientError as e:
            last_error = e
            continue
        except Exception as e:
            return asdict(_fallback_plan(why=f"Unexpected analyzer error: {e}"))

    if last_error is not None:
        reason = f"LLM call failed after {attempts} attempts: {last_error}"
        if _looks_like_gateway_connectivity_issue(str(last_error)):
            reason = (
                f"{reason}. LLM gateway appears unreachable from current node; "
                "please rely on runtime stderr/stdout for troubleshooting and switch to another node."
            )
        return asdict(_fallback_plan(why=reason))

    # Defensive fallback: should be unreachable.
    return asdict(_fallback_plan(why="LLM call failed unexpectedly"))
