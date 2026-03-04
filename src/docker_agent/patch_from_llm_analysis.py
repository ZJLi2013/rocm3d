from __future__ import annotations

"""Apply explicit script patches from llm_analysis JSON.

Flow:
1) Read `llm_analysis_plan.execution_plan` from analysis JSON.
2) If `action != patch_script`, return no-op result (defer to caller policy).
3) For each target script (`install_script` / optional `run_script`), apply patches safely:
   - `replace_line`: requires an exact unique match.
   - `append_block` / `prepend_block`: supports optional anchor by `match`.
4) Return structured apply result (`changed_files`, per-patch records, errors).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional


PatchOp = Literal["replace_line", "append_block", "prepend_block"]
PatchTarget = Literal["install_script", "run_script"]
ExecutionAction = Literal["patch_script", "retry", "need_human"]


@dataclass(frozen=True)
class ScriptPatch:
    op: PatchOp
    target: PatchTarget
    match: str
    content: str


@dataclass(frozen=True)
class PatchApplyRecord:
    target: PatchTarget
    op: PatchOp
    status: Literal["applied", "already_applied", "skipped"]
    detail: str


@dataclass(frozen=True)
class PatchApplyResult:
    action: ExecutionAction
    changed_files: list[str]
    applied: list[PatchApplyRecord]
    errors: list[str]


def _strip_eol(line: str) -> str:
    return line.rstrip("\r\n")


def _infer_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _parse_patches(execution_plan: dict) -> tuple[ExecutionAction, list[ScriptPatch]]:
    action = str(execution_plan.get("action", "need_human"))
    if action not in {"patch_script", "retry", "need_human"}:
        action = "need_human"

    patches: list[ScriptPatch] = []
    for item in execution_plan.get("patches", []) or []:
        op = str(item.get("op", "append_block"))
        target = str(item.get("target", "install_script"))
        if op not in {"replace_line", "append_block", "prepend_block"}:
            continue
        if target not in {"install_script", "run_script"}:
            continue
        patches.append(
            ScriptPatch(
                op=op,  # type: ignore[arg-type]
                target=target,  # type: ignore[arg-type]
                match=str(item.get("match", "")),
                content=str(item.get("content", "")),
            )
        )
    return action, patches


def _apply_replace_line(lines: list[str], patch: ScriptPatch, newline: str) -> tuple[list[str], PatchApplyRecord, bool]:
    match_indices = [i for i, line in enumerate(lines) if _strip_eol(line) == patch.match]
    if len(match_indices) > 1:
        raise ValueError(f"replace_line match is ambiguous: {patch.match!r}")
    if len(match_indices) == 0:
        if any(_strip_eol(line) == patch.content for line in lines):
            return (
                lines,
                PatchApplyRecord(
                    target=patch.target,
                    op=patch.op,
                    status="already_applied",
                    detail="content already exists and match not found",
                ),
                False,
            )
        raise ValueError(f"replace_line match not found: {patch.match!r}")

    idx = match_indices[0]
    new_lines = list(lines)
    new_lines[idx] = patch.content + newline
    return (
        new_lines,
        PatchApplyRecord(
            target=patch.target,
            op=patch.op,
            status="applied",
            detail=f"replaced line at index {idx}",
        ),
        True,
    )


def _find_anchor(lines: list[str], match: str) -> int:
    indices = [i for i, line in enumerate(lines) if _strip_eol(line) == match]
    if len(indices) > 1:
        raise ValueError(f"anchor match is ambiguous: {match!r}")
    if not indices:
        raise ValueError(f"anchor match not found: {match!r}")
    return indices[0]


def _apply_append_or_prepend(lines: list[str], patch: ScriptPatch, newline: str) -> tuple[list[str], PatchApplyRecord, bool]:
    if not patch.content.strip():
        return (
            lines,
            PatchApplyRecord(
                target=patch.target,
                op=patch.op,
                status="skipped",
                detail="empty content",
            ),
            False,
        )

    file_text = "".join(lines)
    if patch.content in file_text:
        return (
            lines,
            PatchApplyRecord(
                target=patch.target,
                op=patch.op,
                status="already_applied",
                detail="content already present",
            ),
            False,
        )

    block_lines = [line + newline for line in patch.content.splitlines()]
    new_lines = list(lines)

    if patch.op == "append_block":
        insert_pos = len(new_lines) if not patch.match else _find_anchor(new_lines, patch.match) + 1
        new_lines[insert_pos:insert_pos] = block_lines
        detail = f"appended block at index {insert_pos}"
    else:
        insert_pos = 0 if not patch.match else _find_anchor(new_lines, patch.match)
        new_lines[insert_pos:insert_pos] = block_lines
        detail = f"prepended block at index {insert_pos}"

    return (
        new_lines,
        PatchApplyRecord(
            target=patch.target,
            op=patch.op,
            status="applied",
            detail=detail,
        ),
        True,
    )


def _apply_patches_to_file(path: Path, patches: list[ScriptPatch]) -> tuple[bool, list[PatchApplyRecord], list[str]]:
    if not path.exists():
        return False, [], [f"target script not found: {path}"]

    original = path.read_text(encoding="utf-8", errors="replace")
    newline = _infer_newline(original)
    lines = original.splitlines(keepends=True)

    records: list[PatchApplyRecord] = []
    errors: list[str] = []
    changed = False

    for patch in patches:
        try:
            if patch.op == "replace_line":
                lines, rec, did_change = _apply_replace_line(lines, patch, newline)
            else:
                lines, rec, did_change = _apply_append_or_prepend(lines, patch, newline)
            records.append(rec)
            changed = changed or did_change
        except Exception as exc:
            errors.append(f"{patch.target}:{patch.op}: {exc}")

    if changed and not errors:
        path.write_text("".join(lines), encoding="utf-8")
    return changed, records, errors


def apply_patches_from_llm_analysis(
    *,
    analysis_json_path: str,
    install_script_path: str,
    run_script_path: Optional[str] = None,
) -> dict:
    """Apply `execution_plan.patches` to install/run scripts.

    This function is generic and schema-driven. It only performs explicit patching
    when `execution_plan.action == "patch_script"`. For other actions (`retry`,
    `need_human`), it returns a no-op result and leaves policy decisions to callers.
    """
    analysis = json.loads(Path(analysis_json_path).read_text(encoding="utf-8"))
    llm_plan = analysis.get("llm_analysis_plan") or {}
    return apply_patches_from_plan(
        llm_plan=llm_plan,
        install_script_path=install_script_path,
        run_script_path=run_script_path,
    )


def apply_patches_from_plan(
    *,
    llm_plan: dict,
    install_script_path: str,
    run_script_path: Optional[str] = None,
) -> dict:
    """Apply patches from an in-memory `llm_analysis_plan` object."""
    execution_plan = llm_plan.get("execution_plan") or {}
    action, parsed_patches = _parse_patches(execution_plan)

    if action != "patch_script":
        return asdict(
            PatchApplyResult(
                action=action,
                changed_files=[],
                applied=[],
                errors=[],
            )
        )

    target_paths: dict[PatchTarget, Path] = {"install_script": Path(install_script_path)}
    if run_script_path:
        target_paths["run_script"] = Path(run_script_path)

    all_records: list[PatchApplyRecord] = []
    all_errors: list[str] = []
    changed_files: list[str] = []

    for target in ("install_script", "run_script"):
        target_typed = target  # keep static type simple
        patches_for_target = [p for p in parsed_patches if p.target == target_typed]
        if not patches_for_target:
            continue
        script_path = target_paths.get(target_typed)  # type: ignore[arg-type]
        if script_path is None:
            all_errors.append(f"missing path for target: {target_typed}")
            continue

        changed, records, errors = _apply_patches_to_file(script_path, patches_for_target)
        all_records.extend(records)
        all_errors.extend(errors)
        if changed:
            changed_files.append(str(script_path))

    return asdict(
        PatchApplyResult(
            action=action,
            changed_files=changed_files,
            applied=all_records,
            errors=all_errors,
        )
    )
