from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model_io import load_bundle, load_checkpoint, predict_rows
from score_rules import best_rows_by_cost, build_cost_summary, evaluate_budget_methods

BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
INTERVENTION_COSTS = [0.0, 0.25, 0.5, 1.0, 2.0]


def compare_methods(dataset_dir: Path, model_paths: list[Path], output_dir: Path) -> Path:
    bundle = load_bundle(dataset_dir)
    rows = bundle["rows"]
    test_rows = rows[rows["split"] == "test"].copy().reset_index(drop=True)
    if test_rows.empty:
        test_rows = rows.copy().reset_index(drop=True)

    method_scores = {
        "delta_q": test_rows["delta_q"].to_numpy(dtype="float32"),
        "action_l2_distance": test_rows["action_l2_distance"].to_numpy(dtype="float32"),
        "mistake_magnitude": test_rows["mistake_magnitude"].to_numpy(dtype="float32"),
    }

    learned_tables = []
    for model_path in model_paths:
        payload = load_checkpoint(model_path)
        prediction_df = predict_rows(payload, bundle, test_rows)
        label = str(payload["model_name"])
        method_scores[label] = prediction_df["pred_intervention_score"].to_numpy(dtype="float32")
        learned_tables.append(prediction_df)

    # We keep the evaluation focused on the reported operating mode:
    # budgeted selection plus cost-adjusted net return.
    budget_summary = evaluate_budget_methods(test_rows, method_scores, BUDGETS)
    cost_summary = build_cost_summary(budget_summary, INTERVENTION_COSTS)
    best_overall = best_rows_by_cost(cost_summary)
    best_learned = best_rows_by_cost(cost_summary, ["direct_no_delta_q", "direct_with_delta_q"])

    output_dir.mkdir(parents=True, exist_ok=True)
    budget_summary.to_csv(output_dir / "budget_summary.csv", index=False)
    cost_summary.to_csv(output_dir / "cost_summary.csv", index=False)
    best_overall.to_csv(output_dir / "best_overall.csv", index=False)
    best_learned.to_csv(output_dir / "best_learned.csv", index=False)
    if learned_tables:
        pd.concat(learned_tables, ignore_index=True).to_csv(output_dir / "learned_predictions.csv", index=False)

    payload = {
        "best_overall": best_overall.to_dict(orient="records"),
        "best_learned": best_learned.to_dict(orient="records"),
    }
    with (output_dir / "comparison.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare delta_q against the direct models.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_methods(args.dataset_dir, args.model, args.output_dir)


if __name__ == "__main__":
    main()
