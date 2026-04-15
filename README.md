# RoboPianist Cost-Aware Intervention

## Research paper

- Paper PDF: [report.pdf](./report.pdf)
- Paper source: [report.tex](./report.tex)

## Repository layout

- `run_curated_pipeline.py`: end-to-end entrypoint
- `build_dataset.py`: curated dataset generation and audit logic
- `fit_model.py`: direct-value model training
- `compare_methods.py`: baseline and learned-model comparison
- `score_rules.py`: budgeted and cost-aware selection rules
- `model_io.py`: feature building, checkpoint save/load, prediction helpers
- `robopianist-rl/`: minimal RL files needed to replay the expert checkpoints

## Requirements

- Python `3.10`
- The five curated checkpoints available locally under `robopianist_rl_runs/`
- System packages needed by RoboPianist and MuJoCo if your machine does not already have them

Install the Python dependencies with:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The checkpoints are not tracked in Git. The pipeline expects these local files:

- `robopianist_rl_runs/SAC-RoboPianist-debug-TwinkleTwinkleLittleStar-v0-42-1775611195.3773801/checkpoint_latest.pkl`
- `robopianist_rl_runs/multi_seed-TwinkleTwinkleLittleStar-seed7-full500k/checkpoint_latest.pkl`
- `robopianist_rl_runs/multi_seed-TwinkleTwinkleLittleStar-seed123-full500k/checkpoint_latest.pkl`
- `robopianist_rl_runs/multi_task-TwinkleTwinkleRousseau-seed42-200kserial/checkpoint_latest.pkl`
- `robopianist_rl_runs/multi_task-CMajorScaleTwoHands-seed42-200kserial/checkpoint_latest.pkl`

## How to run

Run the pipeline:

```bash
python run_curated_pipeline.py --output-root results/curated_pipeline --overwrite
```


## Pipeline steps

1. `build_dataset.py` rebuilds the curated HLPE benchmark from the five final checkpoints.
2. `fit_model.py` trains `direct_no_delta_q` and `direct_with_delta_q`.
3. `compare_methods.py` evaluates those models against `delta_q`, `action_l2_distance`, and `mistake_magnitude`.
4. `run_curated_pipeline.py` writes the final markdown summary under `results/.../summary/summary.md`.

