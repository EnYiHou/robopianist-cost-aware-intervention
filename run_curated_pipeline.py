from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_dataset import build_dataset
from compare_methods import compare_methods
from fit_model import train_one_model

PROJECT_ROOT = Path(__file__).resolve().parent

CURATED_CHECKPOINTS = [
    {
        "label": "twinkle_seed42",
        "path": str(PROJECT_ROOT / "robopianist_rl_runs" / "SAC-RoboPianist-debug-TwinkleTwinkleLittleStar-v0-42-1775611195.3773801" / "checkpoint_latest.pkl"),
    },
    {
        "label": "twinkle_seed7",
        "path": str(PROJECT_ROOT / "robopianist_rl_runs" / "multi_seed-TwinkleTwinkleLittleStar-seed7-full500k" / "checkpoint_latest.pkl"),
    },
    {
        "label": "twinkle_seed123",
        "path": str(PROJECT_ROOT / "robopianist_rl_runs" / "multi_seed-TwinkleTwinkleLittleStar-seed123-full500k" / "checkpoint_latest.pkl"),
    },
    {
        "label": "twinkle_rousseau_seed42",
        "path": str(PROJECT_ROOT / "robopianist_rl_runs" / "multi_task-TwinkleTwinkleRousseau-seed42-200kserial" / "checkpoint_latest.pkl"),
    },
    {
        "label": "cmajor_scale_seed42",
        "path": str(PROJECT_ROOT / "robopianist_rl_runs" / "multi_task-CMajorScaleTwoHands-seed42-200kserial" / "checkpoint_latest.pkl"),
    },
]


def best_method_rows(cost_summary: pd.DataFrame, method: str) -> pd.DataFrame:
    method_rows = cost_summary[cost_summary["method"] == method].copy()
    rows = []
    for cost, group in method_rows.groupby("intervention_cost", dropna=False):
        rows.append(group.sort_values("mean_net_return", ascending=False).iloc[0].to_dict())
    return pd.DataFrame(rows)


def write_summary(dataset_dir: Path, compare_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "dataset_audit.json").open() as handle:
        audit = json.load(handle)
    cost_summary = pd.read_csv(compare_dir / "cost_summary.csv")
    direct_rows = best_method_rows(cost_summary, "direct_with_delta_q")
    delta_rows = best_method_rows(cost_summary, "delta_q")
    merged = direct_rows.merge(delta_rows, on="intervention_cost", suffixes=("_direct", "_delta"))
    merged["gap"] = merged["mean_net_return_direct"] - merged["mean_net_return_delta"]

    positive_cost_rows = merged[merged["intervention_cost"] > 0].copy()
    wins_all_positive_costs = bool((positive_cost_rows["gap"] > 0).all()) if not positive_cost_rows.empty else False

    lines = [
        "# Curated HLPE Summary",
        "",
        "This repo keeps only the curated RoboPianist teacher-intervention pipeline.",
        "",
        "## Dataset",
        "",
        f"- Rows: {audit['rows']}",
        f"- Tasks: {audit['num_tasks']}",
        f"- Positive intervention-value fraction: {audit['positive_value_fraction']:.6f}",
        f"- Recoverable fraction: {audit['recoverable_fraction']:.6f}",
        "",
        "## Main comparison",
        "",
        "```text",
        positive_cost_rows[["intervention_cost", "mean_net_return_direct", "mean_net_return_delta", "gap"]].to_string(index=False),
        "```",
        "",
        f"Claim check: `direct_with_delta_q` beats plain `delta_q` at every positive intervention cost: `{wins_all_positive_costs}`.",
        "",
        f"Dataset: {dataset_dir}",
        f"Comparison: {compare_dir}",
    ]
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def run_pipeline(output_root: Path, anchor_count: int = 16, seed: int = 0, epochs: int = 25, batch_size: int = 128, overwrite: bool = False) -> dict[str, Path]:
    dataset_dir = output_root / "dataset"
    model_dir = output_root / "models"
    compare_dir = output_root / "comparison"
    summary_dir = output_root / "summary"

    build_dataset(
        checkpoints=CURATED_CHECKPOINTS,
        output_dir=dataset_dir,
        anchor_count=anchor_count,
        seed=seed,
        overwrite=overwrite,
    )
    no_delta = train_one_model(dataset_dir, model_dir, include_delta_q=False, seed=seed, epochs=epochs, batch_size=batch_size)
    with_delta = train_one_model(dataset_dir, model_dir, include_delta_q=True, seed=seed, epochs=epochs, batch_size=batch_size)
    compare_methods(dataset_dir, [no_delta, with_delta], compare_dir)
    summary_path = write_summary(dataset_dir, compare_dir, summary_dir)
    return {
        "dataset_dir": dataset_dir,
        "model_dir": model_dir,
        "compare_dir": compare_dir,
        "summary_path": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the curated HLPE pipeline end to end.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "curated_pipeline")
    parser.add_argument("--anchor-count", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        output_root=args.output_root,
        anchor_count=args.anchor_count,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(f"summary: {outputs['summary_path']}")


if __name__ == "__main__":
    main()
