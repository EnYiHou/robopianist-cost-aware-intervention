from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def select_top_budget(scores: np.ndarray, budget: float) -> np.ndarray:
    budget = float(budget)
    if budget <= 0:
        return np.zeros(len(scores), dtype=bool)
    if budget >= 1:
        return np.ones(len(scores), dtype=bool)
    keep = int(np.ceil(len(scores) * budget))
    order = np.argsort(scores)[::-1]
    mask = np.zeros(len(scores), dtype=bool)
    for idx in order[:keep]:
        mask[int(idx)] = True
    return mask


def build_episode_outcomes(rows: pd.DataFrame, method: str, selection_type: str, selection_value: float, intervene_mask: np.ndarray) -> dict[str, float | str]:
    intervene_mask = intervene_mask.astype(bool)
    chosen_rewards = np.where(
        intervene_mask,
        rows["corrected_return_20"].to_numpy(dtype=np.float32),
        rows["uncorrected_return_20"].to_numpy(dtype=np.float32),
    )
    return {
        "method": method,
        "selection_type": selection_type,
        "selection_value": float(selection_value),
        "intervention_rate": float(intervene_mask.mean()),
        "mean_reward": float(np.mean(chosen_rewards)),
    }


def evaluate_budget_methods(rows: pd.DataFrame, method_scores: dict[str, np.ndarray], budgets: Iterable[float]) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    fixed_methods = {
        "always_intervene": np.ones(len(rows), dtype=bool),
        "never_intervene": np.zeros(len(rows), dtype=bool),
    }
    for method, mask in fixed_methods.items():
        records.append(build_episode_outcomes(rows, method, "fixed", 0.0, mask))
    for method, scores in method_scores.items():
        scores = np.asarray(scores, dtype=np.float32)
        for budget in budgets:
            mask = select_top_budget(scores, float(budget))
            records.append(build_episode_outcomes(rows, method, "budget", float(budget), mask))
    return pd.DataFrame(records)


def build_cost_summary(summary: pd.DataFrame, intervention_costs: Iterable[float]) -> pd.DataFrame:
    rows = []
    for _, record in summary.iterrows():
        for cost in intervention_costs:
            mean_net_return = float(record["mean_reward"]) - float(cost) * float(record["intervention_rate"])
            rows.append(
                {
                    "method": record["method"],
                    "selection_type": record["selection_type"],
                    "selection_value": float(record["selection_value"]),
                    "intervention_rate": float(record["intervention_rate"]),
                    "mean_reward": float(record["mean_reward"]),
                    "intervention_cost": float(cost),
                    "mean_net_return": mean_net_return,
                }
            )
    return pd.DataFrame(rows)


def best_rows_by_cost(cost_summary: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    if methods is None:
        filtered = cost_summary.copy()
    else:
        filtered = cost_summary[cost_summary["method"].isin(methods)].copy()
    rows = []
    for cost, group in filtered.groupby("intervention_cost", dropna=False):
        rows.append(group.sort_values("mean_net_return", ascending=False).iloc[0].to_dict())
    return pd.DataFrame(rows)
