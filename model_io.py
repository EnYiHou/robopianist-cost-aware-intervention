from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from build_dataset import load_dataset


class DirectValueModel(nn.Module):
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.gelu(x)
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


@dataclass(frozen=True)
class FeatureBundle:
    matrix: np.ndarray
    feature_spec: dict[str, Any]


def encode_categories(values: pd.Series, vocab: list[str]) -> np.ndarray:
    matrix = np.zeros((len(values), len(vocab)), dtype=np.float32)
    index = {value: i for i, value in enumerate(vocab)}
    for row_index, value in enumerate(values.astype(str).tolist()):
        if value in index:
            matrix[row_index, index[value]] = 1.0
    return matrix


def build_feature_matrix(bundle: dict[str, Any], rows: pd.DataFrame | None = None, include_delta_q: bool = False, feature_spec: dict[str, Any] | None = None) -> FeatureBundle:
    if rows is None:
        rows = bundle["rows"]
    obs = bundle["observations"][rows["obs_index"].to_numpy(dtype=int)]
    expert_actions = bundle["expert_actions"][rows["expert_action_index"].to_numpy(dtype=int)]
    candidate_actions = bundle["candidate_actions"][rows["candidate_action_index"].to_numpy(dtype=int)]
    action_delta = candidate_actions - expert_actions

    if feature_spec is None:
        task_vocab = sorted(bundle["rows"]["task_id"].astype(str).unique().tolist())
        subfamily_vocab = sorted(bundle["rows"]["hlp_subfamily"].fillna("none").astype(str).unique().tolist())
    else:
        task_vocab = list(feature_spec["task_vocab"])
        subfamily_vocab = list(feature_spec["subfamily_vocab"])

    task_features = encode_categories(rows["task_id"].astype(str), task_vocab)
    subfamily_values = rows["hlp_subfamily"].fillna("none").astype(str)
    subfamily_features = encode_categories(subfamily_values, subfamily_vocab)

    scalar_columns = [
        rows["progress"].to_numpy(dtype=np.float32).reshape(-1, 1),
        rows["action_l2_distance"].to_numpy(dtype=np.float32).reshape(-1, 1),
        rows["mistake_magnitude"].to_numpy(dtype=np.float32).reshape(-1, 1),
        rows["burst_length"].to_numpy(dtype=np.float32).reshape(-1, 1),
        rows["support_size"].to_numpy(dtype=np.float32).reshape(-1, 1),
        rows["lag_alpha"].fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1),
    ]
    scalar_names = [
        "progress",
        "action_l2_distance",
        "mistake_magnitude",
        "burst_length",
        "support_size",
        "lag_alpha",
    ]
    if include_delta_q:
        scalar_columns.append(rows["delta_q"].to_numpy(dtype=np.float32).reshape(-1, 1))
        scalar_names.append("delta_q")

    matrix = np.concatenate(
        [
            obs.astype(np.float32),
            expert_actions.astype(np.float32),
            candidate_actions.astype(np.float32),
            action_delta.astype(np.float32),
            *scalar_columns,
            task_features.astype(np.float32),
            subfamily_features.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    # The feature layout is kept explicit on purpose so it's easy to inspect
    # what the model actually sees.
    spec = {
        "include_delta_q": bool(include_delta_q),
        "task_vocab": task_vocab,
        "subfamily_vocab": subfamily_vocab,
        "scalar_names": scalar_names,
        "input_dim": int(matrix.shape[1]),
    }
    return FeatureBundle(matrix=matrix, feature_spec=spec)


def normalize_features(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((matrix - mean) / std).astype(np.float32)


def create_model(hidden_dims: tuple[int, ...] = (256, 256)) -> DirectValueModel:
    return DirectValueModel(hidden_dims=hidden_dims)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_rows(payload: dict[str, Any], bundle: dict[str, Any], rows: pd.DataFrame | None = None) -> pd.DataFrame:
    if rows is None:
        rows = bundle["rows"]
    feature_bundle = build_feature_matrix(
        bundle,
        rows,
        include_delta_q=bool(payload["feature_spec"]["include_delta_q"]),
        feature_spec=payload["feature_spec"],
    )
    matrix = normalize_features(
        feature_bundle.matrix,
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
    )
    model = create_model(tuple(payload["hidden_dims"]))
    values = model.apply({"params": payload["params"]}, jnp.asarray(matrix, dtype=jnp.float32))
    predictions = rows.copy().reset_index(drop=True)
    predictions["pred_intervention_value_20"] = np.asarray(values, dtype=np.float32)
    predictions["pred_intervention_score"] = predictions["pred_intervention_value_20"]
    predictions["include_delta_q"] = bool(payload["feature_spec"]["include_delta_q"])
    predictions["model_name"] = str(payload["model_name"])
    return predictions


def load_bundle(dataset_dir: Path) -> dict[str, Any]:
    return load_dataset(dataset_dir)
