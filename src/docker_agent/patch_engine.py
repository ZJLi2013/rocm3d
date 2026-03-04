from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .patch_from_llm_analysis import apply_patches_from_plan


@dataclass(frozen=True)
class PatchApplyRequest:
    llm_plan: dict
    install_script_path: str
    run_script_path: Optional[str] = None


class PatchEngine:
    """Patch application layer for llm analysis execution plans."""

    def __init__(
        self,
        *,
        apply_fn: Callable[..., dict] = apply_patches_from_plan,
    ) -> None:
        self._apply_fn = apply_fn

    def apply(self, req: PatchApplyRequest) -> dict:
        return self._apply_fn(
            llm_plan=req.llm_plan,
            install_script_path=req.install_script_path,
            run_script_path=req.run_script_path,
        )
