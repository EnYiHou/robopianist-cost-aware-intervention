from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_io import build_feature_matrix, create_model, load_bundle, normalize_features, save_checkpoint

HIDDEN_DIMS = (256, 256)
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
EPOCHS = 25


def prepare_rows(bundle: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = bundle["rows"]
    assert isinstance(rows, pd.DataFrame)
    train_rows = rows[rows["split"] == "train"].copy().reset_index(drop=True)
    val_rows = rows[rows["split"] == "val"].copy().reset_index(drop=True)
    test_rows = rows[rows["split"] == "test"].copy().reset_index(drop=True)
    if train_rows.empty:
        raise ValueError("training split is empty")
    if val_rows.empty:
        val_rows = train_rows.copy().reset_index(drop=True)
    if test_rows.empty:
        test_rows = val_rows.copy().reset_index(drop=True)
    return train_rows, val_rows, test_rows


def compute_norm_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True).astype(np.float32)
    std = matrix.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def iterate_batches(num_rows: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(num_rows)
    start = 0
    while start < num_rows:
        yield order[start : start + batch_size]
        start += batch_size


def train_one_model(dataset_dir: Path, output_dir: Path, include_delta_q: bool, seed: int = 0, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> Path:
    bundle = load_bundle(dataset_dir)
    train_rows, val_rows, test_rows = prepare_rows(bundle)

    train_features = build_feature_matrix(bundle, train_rows, include_delta_q=include_delta_q)
    val_features = build_feature_matrix(bundle, val_rows, include_delta_q=include_delta_q, feature_spec=train_features.feature_spec)
    test_features = build_feature_matrix(bundle, test_rows, include_delta_q=include_delta_q, feature_spec=train_features.feature_spec)

    mean, std = compute_norm_stats(train_features.matrix)
    x_train = normalize_features(train_features.matrix, mean, std)
    x_val = normalize_features(val_features.matrix, mean, std)
    x_test = normalize_features(test_features.matrix, mean, std)
    y_train = train_rows["intervention_value_20"].to_numpy(dtype=np.float32)
    y_val = val_rows["intervention_value_20"].to_numpy(dtype=np.float32)
    y_test = test_rows["intervention_value_20"].to_numpy(dtype=np.float32)

    model = create_model(HIDDEN_DIMS)
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(x_train[:1], dtype=jnp.float32))["params"]
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch):
        def loss_fn(current_params):
            predictions = model.apply({"params": current_params}, x_batch)
            return jnp.mean((predictions - y_batch) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(seed)
    last_loss = 0.0
    for _ in range(epochs):
        # A plain shuffled batch loop is enough here and keeps the training path easy to follow.
        for batch_indices in iterate_batches(len(x_train), batch_size, rng):
            x_batch = jnp.asarray(x_train[batch_indices], dtype=jnp.float32)
            y_batch = jnp.asarray(y_train[batch_indices], dtype=jnp.float32)
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            last_loss = float(loss)

    val_predictions = np.asarray(model.apply({"params": params}, jnp.asarray(x_val, dtype=jnp.float32)), dtype=np.float32)
    test_predictions = np.asarray(model.apply({"params": params}, jnp.asarray(x_test, dtype=jnp.float32)), dtype=np.float32)
    metrics = {
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "test_rows": int(len(test_rows)),
        "final_train_loss": float(last_loss),
        "val_mae": float(mean_absolute_error(y_val, val_predictions)),
        "val_rmse": float(mean_squared_error(y_val, val_predictions) ** 0.5),
        "test_mae": float(mean_absolute_error(y_test, test_predictions)),
        "test_rmse": float(mean_squared_error(y_test, test_predictions) ** 0.5),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "direct_with_delta_q" if include_delta_q else "direct_no_delta_q"
    checkpoint_path = output_dir / model_name / "model_checkpoint.pkl"
    payload = {
        "model_name": model_name,
        "hidden_dims": HIDDEN_DIMS,
        "feature_spec": {
            **train_features.feature_spec,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
        },
        "mean": mean,
        "std": std,
        "params": jax.device_get(params),
        "metrics": metrics,
    }
    save_checkpoint(checkpoint_path, payload)

    pd.DataFrame(
        {
            "truth": y_val,
            "prediction": val_predictions,
        }
    ).to_csv(checkpoint_path.parent / "val_predictions.csv", index=False)
    pd.DataFrame(
        {
            "truth": y_test,
            "prediction": test_predictions,
        }
    ).to_csv(checkpoint_path.parent / "test_predictions.csv", index=False)
    with (checkpoint_path.parent / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the curated direct-value model.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-delta-q", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_one_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        include_delta_q=args.include_delta_q,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
