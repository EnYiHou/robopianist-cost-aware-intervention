from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RL_ROOT = PROJECT_ROOT / "robopianist-rl"
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

import sac
import specs
from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

ROWS_FILENAME = "rows.csv"
METADATA_FILENAME = "metadata.json"
OBSERVATIONS_FILENAME = "observations.npy"
EXPERT_ACTIONS_FILENAME = "expert_actions.npy"
CANDIDATE_ACTIONS_FILENAME = "candidate_actions.npy"
CORRECTED_REWARDS_FILENAME = "corrected_rewards_20.npy"
UNCORRECTED_REWARDS_FILENAME = "uncorrected_rewards_20.npy"
AUDIT_CSV_FILENAME = "dataset_audit.csv"
AUDIT_JSON_FILENAME = "dataset_audit.json"

CURATED_SPECS = {
    "hlp_short_burst_noise": [
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 2},
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 3},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 2},
    ],
    "hlp_persistent_sparse_bias": [
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 3, "support_size": 2},
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 3, "support_size": 3},
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 4, "support_size": 2},
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 4, "support_size": 3},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 3, "support_size": 2},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 3, "support_size": 3},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 4, "support_size": 2},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 4, "support_size": 3},
        {"candidate_severity": 1.5, "requested_scale": 1.5, "burst_length": 3, "support_size": 2},
        {"candidate_severity": 1.5, "requested_scale": 1.5, "burst_length": 3, "support_size": 3},
    ],
    "hlp_sparse_sign_template": [
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 1, "support_size": 3},
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 1, "support_size": 4},
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 2, "support_size": 3},
        {"candidate_severity": 0.5, "requested_scale": 0.5, "burst_length": 2, "support_size": 4},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 1, "support_size": 3},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 1, "support_size": 4},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 2, "support_size": 3},
        {"candidate_severity": 1.0, "requested_scale": 1.0, "burst_length": 2, "support_size": 4},
    ],
    "hlp_lagged_action_error": [
        {"candidate_severity": 0.6, "requested_scale": 0.6, "burst_length": 3, "lag_alpha": 0.6},
        {"candidate_severity": 0.6, "requested_scale": 0.6, "burst_length": 4, "lag_alpha": 0.6},
        {"candidate_severity": 0.6, "requested_scale": 0.6, "burst_length": 5, "lag_alpha": 0.6},
        {"candidate_severity": 0.75, "requested_scale": 0.75, "burst_length": 3, "lag_alpha": 0.75},
        {"candidate_severity": 0.75, "requested_scale": 0.75, "burst_length": 4, "lag_alpha": 0.75},
        {"candidate_severity": 0.75, "requested_scale": 0.75, "burst_length": 5, "lag_alpha": 0.75},
        {"candidate_severity": 0.9, "requested_scale": 0.9, "burst_length": 3, "lag_alpha": 0.9},
        {"candidate_severity": 0.9, "requested_scale": 0.9, "burst_length": 4, "lag_alpha": 0.9},
        {"candidate_severity": 0.9, "requested_scale": 0.9, "burst_length": 5, "lag_alpha": 0.9},
        {"candidate_severity": 0.6, "requested_scale": 0.6, "burst_length": 3, "lag_alpha": 0.6},
        {"candidate_severity": 0.75, "requested_scale": 0.75, "burst_length": 4, "lag_alpha": 0.75},
    ],
}


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    agent_config: sac.SACConfig = sac.SACConfig()


def get_env(args: Args, record_dir: Optional[Path] = None):
    env = suite.load(
        environment_name=args.environment_name,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=args.record_every)
        env = robopianist_wrappers.MidiEvaluationWrapper(environment=env, deque_size=args.record_every)
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(env, num_frames=args.frame_stack, flatten=True)
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env


def stable_seed(*parts: Any) -> int:
    # We keep split assignment and candidate generation stable by hashing labels
    # instead of depending on filesystem ordering.
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def task_id_from_environment_name(environment_name: str) -> str:
    name = environment_name.removeprefix("RoboPianist-debug-")
    return name.removesuffix("-v0")


def rolling_window_sums(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(values) < window:
        return np.zeros((0,), dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32)
    return np.convolve(values.astype(np.float32), kernel, mode="valid")


def compute_recovery_labels(
    corrected_rewards: np.ndarray,
    uncorrected_rewards: np.ndarray,
    recovery_window: int = 5,
    recovery_horizon: int = 20,
    recovery_threshold: float = 1.0,
) -> tuple[int, int | None]:
    corrected_windows = rolling_window_sums(corrected_rewards, recovery_window)
    uncorrected_windows = rolling_window_sums(uncorrected_rewards, recovery_window)
    max_offset = recovery_horizon - recovery_window + 1
    max_offset = min(max_offset, len(corrected_windows), len(uncorrected_windows))
    if max_offset <= 0:
        return 0, None
    corrected_view = corrected_windows[:max_offset]
    uncorrected_view = uncorrected_windows[:max_offset]
    threshold_values = corrected_view * recovery_threshold
    hits = np.flatnonzero(uncorrected_view >= threshold_values)
    if len(hits) == 0:
        return 0, None
    return 1, int(hits[0])


def compute_branch_labels(corrected_rewards: np.ndarray, uncorrected_rewards: np.ndarray) -> dict[str, float | int | None]:
    corrected_rewards = corrected_rewards.astype(np.float32)
    uncorrected_rewards = uncorrected_rewards.astype(np.float32)
    corrected_return_5 = float(np.sum(corrected_rewards[:5]))
    uncorrected_return_5 = float(np.sum(uncorrected_rewards[:5]))
    corrected_return_20 = float(np.sum(corrected_rewards[:20]))
    uncorrected_return_20 = float(np.sum(uncorrected_rewards[:20]))
    recoverable_20, time_to_recovery_20 = compute_recovery_labels(
        corrected_rewards,
        uncorrected_rewards,
    )
    return {
        "corrected_return_5": corrected_return_5,
        "uncorrected_return_5": uncorrected_return_5,
        "harm_5": corrected_return_5 - uncorrected_return_5,
        "corrected_return_20": corrected_return_20,
        "uncorrected_return_20": uncorrected_return_20,
        "intervention_value_20": corrected_return_20 - uncorrected_return_20,
        "recoverable_20": int(recoverable_20),
        "time_to_recovery_20": time_to_recovery_20,
    }


def pad_rewards(rewards: list[float], horizon: int = 20) -> np.ndarray:
    padded = np.zeros((horizon,), dtype=np.float32)
    upto = min(len(rewards), horizon)
    if upto:
        padded[:upto] = np.asarray(rewards[:upto], dtype=np.float32)
    return padded


def assign_split(anchor_id: str, val_fraction: float = 0.15, test_fraction: float = 0.15) -> str:
    # Split by anchor id so all candidate variants from the same anchor stay together.
    digest = hashlib.sha256(anchor_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def normalize_direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid vector norm")
    return (vector / norm).astype(np.float32)


def clip_action(action: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> tuple[np.ndarray, bool]:
    clipped = np.clip(action, minimum, maximum).astype(np.float32)
    changed = bool(not np.allclose(action, clipped))
    return clipped, changed


def build_sparse_template_vector(template_seed: int, action_shape: tuple[int, ...], support_size: int) -> np.ndarray:
    flat_dim = int(np.prod(action_shape))
    support_size = max(1, min(flat_dim, int(support_size)))
    rng = np.random.default_rng(template_seed)
    support = rng.choice(flat_dim, size=support_size, replace=False)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=support_size)
    vector = np.zeros((flat_dim,), dtype=np.float32)
    for idx, value in zip(support, signs):
        vector[int(idx)] = float(value)
    return vector.reshape(action_shape).astype(np.float32)


def args_from_checkpoint_blob(blob: dict[str, Any]) -> Args:
    raw = dict(blob["train_args"])
    raw["agent_config"] = sac.SACConfig(**raw["agent_config"])
    if raw.get("record_dir") is not None:
        raw["record_dir"] = Path(raw["record_dir"])
    return Args(**raw)


def load_agent(checkpoint_path: Path) -> tuple[Any, Any, str, int]:
    with checkpoint_path.open("rb") as handle:
        blob = pickle.load(handle)
    train_args = args_from_checkpoint_blob(blob)
    env = get_env(train_args)
    spec = specs.EnvironmentSpec.make(env)
    agent = sac.SAC.initialize(
        spec=spec,
        config=train_args.agent_config,
        seed=train_args.seed,
        discount=train_args.discount,
    )
    agent = agent.replace(
        actor=agent.actor.replace(params=blob["actor"]),
        critic=agent.critic.replace(params=blob["critic"]),
        target_critic=agent.target_critic.replace(params=blob["target_critic"]),
        temp=agent.temp.replace(params=blob["temp"]),
    )
    environment_name = str(blob["train_args"]["environment_name"])
    step = int(blob.get("step", 0))
    return agent, env, environment_name, step


def min_q_value(agent: Any, obs: np.ndarray, action: np.ndarray) -> float:
    obs_batch = jnp.asarray(obs, dtype=jnp.float32).reshape(1, -1)
    action_batch = jnp.asarray(action, dtype=jnp.float32).reshape(1, -1)
    values = agent.critic.apply_fn(
        {"params": agent.critic.params},
        obs_batch,
        action_batch,
        False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )
    return float(jnp.min(values))


def collect_expert_trace(agent: Any, env: Any) -> dict[str, Any]:
    timestep = env.reset()
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    rewards: list[float] = []
    while True:
        observation = np.asarray(timestep.observation, dtype=np.float32)
        action = agent.eval_actions(observation).astype(np.float32)
        observations.append(observation.copy())
        expert_actions.append(action.copy())
        timestep = env.step(action)
        rewards.append(float(timestep.reward or 0.0))
        if timestep.last():
            break
    return {
        "observations": observations,
        "expert_actions": expert_actions,
        "rewards": rewards,
        "episode_length": len(expert_actions),
        "baseline_reward": float(np.sum(rewards)),
    }


def replay_to_anchor(env: Any, expert_actions: list[np.ndarray], anchor_timestep: int) -> np.ndarray:
    timestep = env.reset()
    for step_index in range(anchor_timestep - 1):
        timestep = env.step(np.asarray(expert_actions[step_index], dtype=np.float32))
    return np.asarray(timestep.observation, dtype=np.float32)


def rollout_branch_from_anchor(
    env: Any,
    agent: Any,
    expert_actions: list[np.ndarray],
    anchor_timestep: int,
    first_action_sequence: np.ndarray,
) -> np.ndarray:
    # Both branches return to expert continuation after the local burst.
    replay_to_anchor(env, expert_actions, anchor_timestep)
    rewards: list[float] = []
    timestep = None
    for action in first_action_sequence:
        timestep = env.step(np.asarray(action, dtype=np.float32))
        rewards.append(float(timestep.reward or 0.0))
        if timestep.last():
            return pad_rewards(rewards, 20)
    if timestep is None:
        raise ValueError("missing branch action")
    while len(rewards) < 20 and not timestep.last():
        observation = np.asarray(timestep.observation, dtype=np.float32)
        action = agent.eval_actions(observation).astype(np.float32)
        timestep = env.step(action)
        rewards.append(float(timestep.reward or 0.0))
    return pad_rewards(rewards, 20)


def sample_anchor_timesteps(episode_length: int, anchor_count: int, min_timestep: int, required_future_steps: int, rng: np.random.Generator) -> list[int]:
    max_timestep = episode_length - required_future_steps + 1
    if max_timestep < min_timestep:
        raise ValueError("no valid anchor timesteps remain")
    candidates = np.arange(min_timestep, max_timestep + 1, dtype=int)
    if len(candidates) <= anchor_count:
        return [int(value) for value in candidates]
    selected = rng.choice(candidates, size=anchor_count, replace=False)
    values = [int(value) for value in selected.tolist()]
    values.sort()
    return values


def generate_curated_candidates(
    expert_actions_future: list[np.ndarray],
    previous_expert_action: np.ndarray,
    action_minimum: np.ndarray,
    action_maximum: np.ndarray,
    base_scale: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if not expert_actions_future:
        raise ValueError("need expert future actions")
    first_expert_action = np.asarray(expert_actions_future[0], dtype=np.float32)
    action_shape = tuple(first_expert_action.shape)
    candidates: list[dict[str, Any]] = []
    for subfamily, raw_specs in CURATED_SPECS.items():
        for candidate_index, raw_spec in enumerate(raw_specs):
            spec = dict(raw_spec)
            requested_scale = float(spec["requested_scale"]) * base_scale
            burst_length = int(spec["burst_length"])
            support_size = int(spec.get("support_size", 0))
            lag_alpha = float(spec.get("lag_alpha", np.nan))
            sequence: list[np.ndarray] = []
            clipped_any = False
            template_seed = int(rng.integers(0, np.iinfo(np.uint32).max))

            if subfamily == "hlp_short_burst_noise":
                # Reuse one direction and add a bit of jitter so the burst feels structured,
                # not just like independent white noise every step.
                base_rng = np.random.default_rng(template_seed)
                base_direction = normalize_direction(base_rng.normal(0.0, 1.0, size=action_shape).astype(np.float32))
                for step_index in range(burst_length):
                    step_rng = np.random.default_rng(template_seed + step_index + 1)
                    jitter = step_rng.normal(
                        0.0,
                        max(1e-6, requested_scale * 0.2),
                        size=action_shape,
                    ).astype(np.float32)
                    noise = base_direction * requested_scale + jitter
                    unclipped = np.asarray(expert_actions_future[step_index], dtype=np.float32) + noise
                    candidate_action, clipped = clip_action(unclipped, action_minimum, action_maximum)
                    sequence.append(candidate_action)
                    clipped_any = clipped_any or clipped
            elif subfamily == "hlp_persistent_sparse_bias":
                template_seed = 7103 + 997 * candidate_index
                sparse_vector = build_sparse_template_vector(template_seed, action_shape, support_size)
                bias = normalize_direction(sparse_vector) * requested_scale
                for step_index in range(burst_length):
                    unclipped = np.asarray(expert_actions_future[step_index], dtype=np.float32) + bias
                    candidate_action, clipped = clip_action(unclipped, action_minimum, action_maximum)
                    sequence.append(candidate_action)
                    clipped_any = clipped_any or clipped
            elif subfamily == "hlp_sparse_sign_template":
                template_seed = 12011 + 577 * candidate_index
                sparse_vector = build_sparse_template_vector(template_seed, action_shape, support_size)
                template = normalize_direction(sparse_vector) * requested_scale
                for step_index in range(burst_length):
                    unclipped = np.asarray(expert_actions_future[step_index], dtype=np.float32) + template
                    candidate_action, clipped = clip_action(unclipped, action_minimum, action_maximum)
                    sequence.append(candidate_action)
                    clipped_any = clipped_any or clipped
            elif subfamily == "hlp_lagged_action_error":
                previous_action = np.asarray(previous_expert_action, dtype=np.float32).copy()
                for step_index in range(burst_length):
                    expert_step_action = np.asarray(expert_actions_future[step_index], dtype=np.float32)
                    unclipped = lag_alpha * previous_action + (1.0 - lag_alpha) * expert_step_action
                    candidate_action, clipped = clip_action(unclipped.astype(np.float32), action_minimum, action_maximum)
                    sequence.append(candidate_action)
                    previous_action = candidate_action
                    clipped_any = clipped_any or clipped
            else:
                raise ValueError(f"unsupported subfamily: {subfamily}")

            candidate_sequence = np.stack(sequence, axis=0).astype(np.float32)
            first_candidate = candidate_sequence[0]
            candidates.append(
                {
                    "candidate_family": "human_like_proxy",
                    "hlp_subfamily": subfamily,
                    "candidate_template_id": f"{subfamily}-{candidate_index}",
                    "candidate_severity": float(spec["candidate_severity"]),
                    "mistake_magnitude": float(requested_scale),
                    "candidate_action": first_candidate,
                    "candidate_action_sequence": candidate_sequence,
                    "candidate_seed": int(template_seed),
                    "mistake_action_clipped": clipped_any,
                    "action_l2_distance": float(np.linalg.norm(first_candidate - first_expert_action)),
                    "burst_length": burst_length,
                    "support_size": support_size,
                    "lag_alpha": lag_alpha,
                }
            )
    return candidates


def build_audit_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_columns = ["task_id", "candidate_family", "hlp_subfamily"]
    summary_rows = []
    for group_keys, group in rows.groupby(group_columns, dropna=False):
        task_id, candidate_family, hlp_subfamily = group_keys
        summary_rows.append(
            {
                "task_id": str(task_id),
                "candidate_family": str(candidate_family),
                "hlp_subfamily": str(hlp_subfamily),
                "rows": int(len(group)),
                "recoverable_fraction": float(group["recoverable_20"].mean()),
                "positive_harm_fraction": float((group["harm_5"] > 0).mean()),
                "positive_value_fraction": float((group["intervention_value_20"] > 0).mean()),
            }
        )
    audit_json = {
        "rows": int(len(rows)),
        "num_tasks": int(rows["task_id"].nunique()),
        "recoverable_fraction": float(rows["recoverable_20"].mean()),
        "positive_harm_fraction": float((rows["harm_5"] > 0).mean()),
        "positive_value_fraction": float((rows["intervention_value_20"] > 0).mean()),
        "split_counts": {str(key): int(value) for key, value in rows["split"].value_counts().sort_index().items()},
        "status": "passed",
    }
    return pd.DataFrame(summary_rows), audit_json


def check_dataset(rows: pd.DataFrame, corrected_rewards: np.ndarray, uncorrected_rewards: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rows.empty:
        raise ValueError("dataset is empty")
    for array_name, array in [("corrected_rewards", corrected_rewards), ("uncorrected_rewards", uncorrected_rewards)]:
        if not np.isfinite(array).all():
            raise ValueError(f"{array_name} contains invalid values")
    for column in ["delta_q", "action_l2_distance", "mistake_magnitude", "harm_5", "intervention_value_20", "recoverable_20"]:
        if rows[column].isna().any():
            raise ValueError(f"column has missing values: {column}")
    leaking = rows.groupby("anchor_id")["split"].nunique()
    if (leaking > 1).any():
        raise ValueError("anchor leakage across splits")
    for row_index in range(len(rows)):
        # Recompute the supervision labels from the stored traces and fail early if they drift.
        labels = compute_branch_labels(corrected_rewards[row_index], uncorrected_rewards[row_index])
        for key in ["corrected_return_5", "uncorrected_return_5", "harm_5", "corrected_return_20", "uncorrected_return_20", "intervention_value_20"]:
            if not np.isclose(float(rows.iloc[row_index][key]), float(labels[key]), atol=1e-5):
                raise ValueError(f"label mismatch for {key} at row {row_index}")
    audit_df, audit_json = build_audit_rows(rows)
    return audit_df, audit_json


def save_dataset(
    output_dir: Path,
    rows: pd.DataFrame,
    metadata: dict[str, Any],
    observations: np.ndarray,
    expert_actions: np.ndarray,
    candidate_actions: np.ndarray,
    corrected_rewards: np.ndarray,
    uncorrected_rewards: np.ndarray,
    audit_df: pd.DataFrame,
    audit_json: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    with (output_dir / METADATA_FILENAME).open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    np.save(output_dir / OBSERVATIONS_FILENAME, observations.astype(np.float32))
    np.save(output_dir / EXPERT_ACTIONS_FILENAME, expert_actions.astype(np.float32))
    np.save(output_dir / CANDIDATE_ACTIONS_FILENAME, candidate_actions.astype(np.float32))
    np.save(output_dir / CORRECTED_REWARDS_FILENAME, corrected_rewards.astype(np.float32))
    np.save(output_dir / UNCORRECTED_REWARDS_FILENAME, uncorrected_rewards.astype(np.float32))
    audit_df.to_csv(output_dir / AUDIT_CSV_FILENAME, index=False)
    with (output_dir / AUDIT_JSON_FILENAME).open("w") as handle:
        json.dump(audit_json, handle, indent=2, sort_keys=True)


def load_dataset(dataset_dir: Path) -> dict[str, Any]:
    rows = pd.read_csv(dataset_dir / ROWS_FILENAME)
    with (dataset_dir / METADATA_FILENAME).open() as handle:
        metadata = json.load(handle)
    return {
        "rows": rows,
        "metadata": metadata,
        "observations": np.load(dataset_dir / OBSERVATIONS_FILENAME),
        "expert_actions": np.load(dataset_dir / EXPERT_ACTIONS_FILENAME),
        "candidate_actions": np.load(dataset_dir / CANDIDATE_ACTIONS_FILENAME),
        "corrected_rewards_20": np.load(dataset_dir / CORRECTED_REWARDS_FILENAME),
        "uncorrected_rewards_20": np.load(dataset_dir / UNCORRECTED_REWARDS_FILENAME),
    }


def build_dataset(checkpoints: list[dict[str, str]], output_dir: Path, anchor_count: int = 16, min_timestep: int = 50, seed: int = 0, base_scale: float = 0.75, overwrite: bool = False) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    expert_actions_store: list[np.ndarray] = []
    candidate_actions_store: list[np.ndarray] = []
    corrected_rewards_store: list[np.ndarray] = []
    uncorrected_rewards_store: list[np.ndarray] = []

    required_future_steps = 20
    for checkpoint_spec in checkpoints:
        checkpoint_path = Path(checkpoint_spec["path"])
        label = checkpoint_spec["label"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        agent, env, environment_name, checkpoint_step = load_agent(checkpoint_path)
        action_spec = env.action_spec()
        action_minimum = np.asarray(action_spec.minimum, dtype=np.float32)
        action_maximum = np.asarray(action_spec.maximum, dtype=np.float32)
        expert_trace = collect_expert_trace(agent, env)
        task_id = task_id_from_environment_name(environment_name)
        rng = np.random.default_rng(stable_seed(seed, label, checkpoint_path))
        anchors = sample_anchor_timesteps(
            int(expert_trace["episode_length"]),
            anchor_count,
            min_timestep,
            required_future_steps,
            rng,
        )

        for anchor_timestep in anchors:
            # One anchor state becomes many local intervention questions through the HLPE family.
            anchor_index = anchor_timestep - 1
            observation = np.asarray(expert_trace["observations"][anchor_index], dtype=np.float32)
            expert_action = np.asarray(expert_trace["expert_actions"][anchor_index], dtype=np.float32)
            previous_action = np.asarray(expert_trace["expert_actions"][max(anchor_index - 1, 0)], dtype=np.float32)
            future_actions = [
                np.asarray(action, dtype=np.float32)
                for action in expert_trace["expert_actions"][anchor_index : anchor_index + required_future_steps]
            ]
            candidate_rng = np.random.default_rng(stable_seed(label, anchor_timestep, seed))
            candidates = generate_curated_candidates(
                future_actions,
                previous_action,
                action_minimum,
                action_maximum,
                base_scale,
                candidate_rng,
            )

            for candidate in candidates:
                anchor_id = f"{label}:{task_id}:t{anchor_timestep}"
                split = assign_split(anchor_id)
                corrected_sequence = np.expand_dims(expert_action, axis=0).astype(np.float32)
                uncorrected_sequence = np.asarray(candidate["candidate_action_sequence"], dtype=np.float32)
                corrected_rewards = rollout_branch_from_anchor(env, agent, expert_trace["expert_actions"], anchor_timestep, corrected_sequence)
                uncorrected_rewards = rollout_branch_from_anchor(env, agent, expert_trace["expert_actions"], anchor_timestep, uncorrected_sequence)
                labels = compute_branch_labels(corrected_rewards, uncorrected_rewards)

                obs_index = len(observations)
                action_index = len(expert_actions_store)
                candidate_index = len(candidate_actions_store)
                trace_index = len(corrected_rewards_store)
                observations.append(observation.copy())
                expert_actions_store.append(expert_action.copy())
                candidate_actions_store.append(np.asarray(candidate["candidate_action"], dtype=np.float32).copy())
                corrected_rewards_store.append(corrected_rewards)
                uncorrected_rewards_store.append(uncorrected_rewards)

                delta_q = min_q_value(agent, observation, expert_action) - min_q_value(
                    agent, observation, np.asarray(candidate["candidate_action"], dtype=np.float32)
                )
                row = {
                    "row_id": trace_index,
                    "anchor_id": anchor_id,
                    "split": split,
                    "environment_name": environment_name,
                    "task_id": task_id,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step,
                    "source_checkpoint_label": label,
                    "anchor_timestep": anchor_timestep,
                    "episode_length": int(expert_trace["episode_length"]),
                    "progress": float(anchor_timestep / expert_trace["episode_length"]),
                    "candidate_family": candidate["candidate_family"],
                    "hlp_subfamily": candidate["hlp_subfamily"],
                    "candidate_template_id": candidate["candidate_template_id"],
                    "candidate_severity": candidate["candidate_severity"],
                    "candidate_seed": candidate["candidate_seed"],
                    "mistake_action_clipped": candidate["mistake_action_clipped"],
                    "burst_length": candidate["burst_length"],
                    "support_size": candidate["support_size"],
                    "lag_alpha": candidate["lag_alpha"],
                    "obs_index": obs_index,
                    "expert_action_index": action_index,
                    "candidate_action_index": candidate_index,
                    "trace_index": trace_index,
                    "delta_q": float(delta_q),
                    "action_l2_distance": float(candidate["action_l2_distance"]),
                    "mistake_magnitude": float(candidate["mistake_magnitude"]),
                }
                row.update(labels)
                rows.append(row)

    rows_df = pd.DataFrame(rows)
    observations_array = np.stack(observations, axis=0).astype(np.float32)
    expert_actions_array = np.stack(expert_actions_store, axis=0).astype(np.float32)
    candidate_actions_array = np.stack(candidate_actions_store, axis=0).astype(np.float32)
    corrected_rewards_array = np.stack(corrected_rewards_store, axis=0).astype(np.float32)
    uncorrected_rewards_array = np.stack(uncorrected_rewards_store, axis=0).astype(np.float32)
    audit_df, audit_json = check_dataset(rows_df, corrected_rewards_array, uncorrected_rewards_array)
    metadata = {
        "anchor_count": int(anchor_count),
        "min_timestep": int(min_timestep),
        "seed": int(seed),
        "base_scale": float(base_scale),
        "checkpoints": checkpoints,
    }
    save_dataset(
        output_dir,
        rows_df,
        metadata,
        observations_array,
        expert_actions_array,
        candidate_actions_array,
        corrected_rewards_array,
        uncorrected_rewards_array,
        audit_df,
        audit_json,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the curated HLPE dataset.")
    parser.add_argument("--checkpoint", action="append", default=[], help="label=path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=16)
    parser.add_argument("--min-timestep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-scale", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = []
    for raw in args.checkpoint:
        if "=" not in raw:
            raise ValueError("checkpoint entries must look like label=/path/to/checkpoint.pkl")
        label, path = raw.split("=", 1)
        checkpoints.append({"label": label, "path": path})
    if not checkpoints:
        raise ValueError("pass at least one --checkpoint")
    build_dataset(
        checkpoints=checkpoints,
        output_dir=args.output_dir,
        anchor_count=args.anchor_count,
        min_timestep=args.min_timestep,
        seed=args.seed,
        base_scale=args.base_scale,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
