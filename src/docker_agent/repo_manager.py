from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CloneResult:
    repo_dir: Path
    submodules_skipped: bool = False  # True when recursive clone fell back to shallow-only


class RepoManager:
    """
    RepoManager:
    - git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 (default).
    - If any submodule URL is unreachable (deleted/moved repo), falls back to cloning
      only the main repo (no submodules). The install script is responsible for any
      submodule init that may still be needed.
    """

    def clone(
        self, repo_url: str, dst_dir: Path, depth: int = 1, recursive: bool = True
    ) -> CloneResult:
        dst_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["git", "clone", "--depth", str(depth)]
        if recursive:
            cmd.extend(["--recurse-submodules", "--shallow-submodules", "--jobs", "4"])
        cmd.extend([repo_url, str(dst_dir)])
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = "\n".join([x for x in [stdout, stderr] if x])

            # Submodule URL unreachable (deleted/moved repo) — retry without submodules
            # so the main repo is still available for the install script.
            _submodule_failure = recursive and any(
                kw in msg.lower()
                for kw in ("submodule", "repository not found", "fatal: clone of")
            )
            if _submodule_failure:
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                cmd_shallow = ["git", "clone", "--depth", str(depth), repo_url, str(dst_dir)]
                proc2 = subprocess.run(cmd_shallow, capture_output=True, text=True)
                if proc2.returncode == 0:
                    return CloneResult(repo_dir=dst_dir, submodules_skipped=True)
                msg2 = "\n".join(
                    [x for x in [(proc2.stdout or "").strip(), (proc2.stderr or "").strip()] if x]
                )
                raise RuntimeError(f"git clone failed: {msg2}")

            raise RuntimeError(f"git clone failed: {msg}")

        return CloneResult(repo_dir=dst_dir)
