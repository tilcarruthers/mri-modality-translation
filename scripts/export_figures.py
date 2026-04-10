from __future__ import annotations

import argparse
from pathlib import Path

from mri_translation.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy selected run figures into reports/figures.")
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Run directory under outputs/runs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    reports_dir = Path("reports/figures")
    ensure_dir(reports_dir)

    for name in ["training_history.png", "prediction_grid.png", "metrics.json"]:
        src = run_dir / name
        if src.exists():
            dst = reports_dir / f"{run_dir.name}_{name}"
            dst.write_bytes(src.read_bytes())
            print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()
