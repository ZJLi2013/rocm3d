from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .agent import DockerAgent, BuildRequest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="docker_agent",
        description="Prepare repo (clone) and optionally run container (Phase 2': base image + install + run).",
    )
    p.add_argument("--repo_url", required=True, help="Git repository URL")
    p.add_argument(
        "--discard-workspace",
        action="store_true",
        help="Do not keep workspace after run (default: keep)",
    )
    p.add_argument(
        "--base-image",
        default=None,
        help="Base image for run step (required if --run-cmd is set)",
    )
    p.add_argument(
        "--install-script",
        default=None,
        metavar="PATH",
        help="Install-phase script path. Copied into repo and run in first container, then commit.",
    )
    p.add_argument(
        "--run-script",
        default=None,
        metavar="PATH",
        help="Run/sample-phase script path. Run in second container from saved image. Optional.",
    )
    p.add_argument(
        "--run-cmd",
        default=None,
        help="Sample run command (one-phase simple case, or second phase when no --run-script).",
    )
    p.add_argument(
        "--install-cmd",
        default=None,
        help="Command to run before --run-cmd (simple case only). Ignored if --install-script is set.",
    )
    p.add_argument(
        "--run-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for container run (default: 3600)",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Write result JSON to this file (in addition to stdout). Always generated when run step is used.",
    )
    p.add_argument(
        "--no-commit-image",
        action="store_true",
        help="Do not save container as image (autorun/<repo>) after successful install (default: commit when install succeeds).",
    )
    p.add_argument(
        "--enable-llm-analysis",
        action="store_true",
        help="Enable LLM failure analysis plan output (dry-run, no script patch).",
    )
    p.add_argument(
        "--llm-timeout-sec",
        type=float,
        default=90.0,
        help="Timeout in seconds for one LLM analysis request (default: 90).",
    )
    p.add_argument(
        "--auto-patch-on-fail",
        action="store_true",
        help="When install fails, auto-apply LLM patches and retry install loop.",
    )
    p.add_argument(
        "--max-auto-patch-retries",
        type=int,
        default=3,
        help="Max install retries in auto patch loop after initial failure (default: 3).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    req = BuildRequest(repo_url=args.repo_url, base_image=args.base_image or None)
    agent = DockerAgent()
    keep_workspace = not args.discard_workspace
    script_from_host = (args.install_script or "").strip() or None
    if script_from_host:
        script_from_host = str(Path(script_from_host).resolve())
    run_script_from_host = (args.run_script or "").strip() or None
    if run_script_from_host:
        run_script_from_host = str(Path(run_script_from_host).resolve())
    run_cmd = args.run_cmd
    install_cmd = None if script_from_host else (args.install_cmd or None)
    result = agent.build_image(
        req,
        keep_workspace=keep_workspace,
        script_from_host=script_from_host,
        run_script_from_host=run_script_from_host,
        run_cmd=run_cmd,
        install_cmd=install_cmd,
        run_timeout_sec=args.run_timeout,
        commit_image=not args.no_commit_image,
        enable_llm_analysis=(args.enable_llm_analysis or args.auto_patch_on_fail),
        llm_timeout_sec=args.llm_timeout_sec,
        auto_patch_on_fail=args.auto_patch_on_fail,
        max_auto_patch_retries=args.max_auto_patch_retries,
    )
    # 容器内脚本的 stdout/stderr 打到 stderr，便于终端查看；JSON 打 stdout，便于重定向
    if result.install_stdout is not None or result.install_stderr is not None:
        if result.install_stdout:
            print("[docker_agent] --- install stdout ---", file=sys.stderr)
            print(result.install_stdout, end="" if result.install_stdout.endswith("\n") else "\n", file=sys.stderr)
        if result.install_stderr:
            print("[docker_agent] --- install stderr ---", file=sys.stderr)
            print(result.install_stderr, end="" if result.install_stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.run_stdout is not None or result.run_stderr is not None:
        if result.run_stdout:
            print("[docker_agent] --- run stdout ---", file=sys.stderr)
            print(result.run_stdout, end="" if result.run_stdout.endswith("\n") else "\n", file=sys.stderr)
        if result.run_stderr:
            print("[docker_agent] --- run stderr ---", file=sys.stderr)
            print(result.run_stderr, end="" if result.run_stderr.endswith("\n") else "\n", file=sys.stderr)
    j = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    print(j)
    if args.output:
        Path(args.output).write_text(j, encoding="utf-8")
    if result.status != "success":
        return 1
    if result.run_exit_code is not None and result.run_exit_code != 0:
        return 1
    if result.run_exit_code is None and result.install_exit_code is not None and result.install_exit_code != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
