"""
Run install or sample container. Uses docker-py; returns exit_code, stdout, stderr, timed_out.
"""
from __future__ import annotations

import concurrent.futures
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import docker

DEFAULT_COMMIT_TIMEOUT_SEC = 1200


@dataclass
class RunResult:
    """Result of run_install_container or run_sample_container."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    saved_image: Optional[str] = None  # only set for install when commit_image_as was used and install succeeded


def _run_impl(
    image: str,
    volumes: Dict[str, Dict[str, Any]],
    shell_cmd: str,
    work_dir: str = "/workspace",
    timeout_sec: int = 3600,
    commit_image_as: Optional[str] = None,
) -> RunResult:
    """Create container, run, optionally commit on success, remove. Shared by install and sample."""
    client = docker.from_env()
    # Mount ROCm GPU devices so torch.cuda.is_available() works inside the container.
    # Falls back gracefully if /dev/kfd or /dev/dri are absent on the host.
    import os as _os
    _rocm_devices: list[str] = []
    if _os.path.exists("/dev/kfd"):
        _rocm_devices.append("/dev/kfd:/dev/kfd:rwm")
    if _os.path.exists("/dev/dri"):
        _rocm_devices.append("/dev/dri:/dev/dri:rwm")
    container = client.containers.create(
        image,
        command=["bash", "-c", shell_cmd],
        volumes=volumes,
        working_dir=work_dir,
        detach=True,
        devices=_rocm_devices if _rocm_devices else None,
        group_add=["video"] if _rocm_devices else None,
        shm_size="32g",
    )
    container.start()

    timed_out = False
    exit_code: int = -1
    saved_image: Optional[str] = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(container.wait)
            try:
                wait_result = fut.result(timeout=timeout_sec)
                exit_code = int(wait_result["StatusCode"])
            except concurrent.futures.TimeoutError:
                timed_out = True
                container.stop(timeout=10)
                wait_result = container.wait()
                exit_code = int(wait_result["StatusCode"])
    finally:
        try:
            out_b, err_b = b"", b""
            try:
                for chunk in container.logs(stdout=True, stderr=True, stream=True, demux=True):
                    if isinstance(chunk, (list, tuple)) and len(chunk) == 2:
                        out_b += chunk[0] or b""
                        err_b += chunk[1] or b""
                    elif isinstance(chunk, bytes):
                        out_b += chunk
            except (TypeError, AttributeError, StopIteration):
                raw = container.logs(stdout=True, stderr=True)
                if isinstance(raw, bytes):
                    out_b = raw
                elif isinstance(raw, (list, tuple)) and len(raw) == 2:
                    out_b = raw[0] or b""
                    err_b = raw[1] or b""
            stdout_s = (out_b or b"").decode("utf-8", errors="replace")
            stderr_s = (err_b or b"").decode("utf-8", errors="replace")
        except Exception:
            stdout_s = ""
            stderr_s = ""
        if commit_image_as and exit_code == 0 and not timed_out:
            try:
                # Use a longer client timeout for image commit; large layers often exceed docker-py default.
                commit_client = docker.from_env(timeout=DEFAULT_COMMIT_TIMEOUT_SEC)
                try:
                    commit_client.api.commit(
                        container=container.id,
                        repository=commit_image_as,
                        tag="latest",
                    )
                finally:
                    close = getattr(commit_client, "close", None)
                    if callable(close):
                        close()
                saved_image = f"{commit_image_as}:latest"
            except Exception as e:
                print(f"[docker_agent] commit_image failed: {e}", file=sys.stderr, flush=True)
        try:
            container.remove(force=True)
        except Exception:
            pass

    return RunResult(
        exit_code=exit_code,
        stdout=stdout_s,
        stderr=stderr_s,
        timed_out=timed_out,
        saved_image=saved_image,
    )


def run_install_container(
    image: str,
    repo_path: str,
    *,
    script_path: Optional[str] = None,
    install_cmd: Optional[str] = None,
    run_cmd: Optional[str] = None,
    work_dir: str = "/workspace",
    timeout_sec: int = 3600,
    commit_image_as: Optional[str] = None,
) -> RunResult:
    """
    Install phase: mount repo at work_dir, run install script or commands; optionally commit on success.
    """
    host_path = str(Path(repo_path).resolve())
    volumes = {host_path: {"bind": work_dir, "mode": "rw"}}
    if script_path:
        shell_cmd = f"cd {work_dir} && bash {script_path}"
    else:
        parts = []
        if install_cmd:
            parts.append(f"({install_cmd})")
        if run_cmd:
            parts.append(f"({run_cmd})")
        if not parts:
            parts.append("true")
        shell_cmd = " && ".join(parts)
    return _run_impl(
        image,
        volumes,
        shell_cmd,
        work_dir=work_dir,
        timeout_sec=timeout_sec,
        commit_image_as=commit_image_as,
    )


def run_sample_container(
    image: str,
    *,
    run_script_host_path: Optional[str] = None,
    run_cmd: Optional[str] = None,
    work_dir: str = "/workspace",
    timeout_sec: int = 3600,
) -> RunResult:
    """
    Sample phase: run from saved image (e.g. autorun/naver-dust3r). Either mount a script or run a single command.
    """
    if run_script_host_path:
        script_host = str(Path(run_script_host_path).resolve())
        volumes = {script_host: {"bind": f"{work_dir}/run_sample.sh", "mode": "ro"}}
        shell_cmd = f"bash {work_dir}/run_sample.sh"
    elif run_cmd:
        volumes = {}
        shell_cmd = run_cmd
    else:
        raise ValueError("run_sample_container requires run_script_host_path or run_cmd")
    return _run_impl(
        image,
        volumes,
        shell_cmd,
        work_dir=work_dir,
        timeout_sec=timeout_sec,
        commit_image_as=None,
    )
