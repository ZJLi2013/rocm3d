from __future__ import annotations

import argparse
import json
from pathlib import Path

from docker_agent.error_classifier import classify_failure
from docker_agent.llm_log_analyzer import analyze_failure_with_llm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LLM analysis plan from docker_agent result JSON.")
    p.add_argument("--input-json", required=True, help="Path to input result JSON.")
    p.add_argument("--output-json", required=True, help="Path to output analysis JSON.")
    p.add_argument("--repo-url", required=True, help="Repository URL for context.")
    p.add_argument("--base-image", default=None, help="Base image name for context.")
    p.add_argument("--stage", choices=["install", "run"], default="install")
    p.add_argument("--script-path", default=None, help="Optional install/run script path for context.")
    p.add_argument("--timeout-sec", type=float, default=90.0, help="LLM timeout in seconds.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if args.stage == "install":
        exit_code = data.get("install_exit_code")
        timed_out = False
        stdout = data.get("install_stdout")
        stderr = data.get("install_stderr")
    else:
        exit_code = data.get("run_exit_code")
        timed_out = bool(data.get("run_timed_out"))
        stdout = data.get("run_stdout")
        stderr = data.get("run_stderr")

    classification = classify_failure(
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
    )

    script_text = None
    if args.script_path:
        sp = Path(args.script_path)
        if sp.exists():
            script_text = sp.read_text(encoding="utf-8", errors="replace")

    plan = analyze_failure_with_llm(
        stage=args.stage,
        repo_url=args.repo_url,
        base_image=args.base_image,
        classification=classification,
        stdout=stdout,
        stderr=stderr,
        script_text=script_text,
        timeout_sec=args.timeout_sec,
    )

    out = {
        "source_json": str(input_path),
        "stage": args.stage,
        "failure_type": classification.failure_type,
        "retryable": classification.retryable,
        "classifier_confidence": classification.confidence,
        "llm_analysis_plan": plan,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
