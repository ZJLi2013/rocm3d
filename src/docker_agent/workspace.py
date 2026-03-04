from __future__ import annotations
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Workspace:
    """
    A per-run workspace directory.

    For MVP, we only guarantee:
    - create a fresh temp directory each run
    - cleanup removes it unless keep=True
    """

    path: Path
    keep: bool = False

    def cleanup(self) -> None:
        if self.keep:
            return
        shutil.rmtree(self.path, ignore_errors=True)


class WorkspaceManager:
    def __init__(self, root_dir: Optional[str] = None) -> None:
        """
        root_dir: optional directory under which the workspace is created.
        If None, uses system temp directory.
        """
        self._root_dir = root_dir

    def create(self, keep: bool = False) -> Workspace:
        p = Path(
            tempfile.mkdtemp(prefix="docker_agent_", dir=self._root_dir)  # type: ignore[arg-type]
        )
        return Workspace(path=p, keep=keep)
