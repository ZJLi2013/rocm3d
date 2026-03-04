from __future__ import annotations

import argparse
import json
from pathlib import Path

from docker_agent.patch_from_llm_analysis import apply_patches_from_llm_analysis


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply llm_analysis_plan.execution_plan patches to scripts.")
    p.add_argument("--analysis-json", required=True, help="Path to llm analysis json file.")
    p.add_argument("--install-script-path", required=True, help="Install script path to patch.")
    p.add_argument("--run-script-path", default=None, help="Optional run script path to patch.")
    p.add_argument("--output-json", default=None, help="Optional path to write apply result json.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = apply_patches_from_llm_analysis(
        analysis_json_path=args.analysis_json,
        install_script_path=args.install_script_path,
        run_script_path=args.run_script_path,
    )

    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(str(out_path))
    else:
        print(out)

    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
