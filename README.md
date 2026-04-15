# RoboPianist Cost-Aware Intervention

Minimal research repo for the curated RoboPianist teacher-intervention pipeline.

The repo keeps one reproducible path only:
- build the curated HLPE dataset from the final checkpoints
- train the retained direct intervention models
- compare them against `delta_q`
- write a short summary with the headline cost-adjusted result

## Research paper

- Paper: [report.md](./report.md)

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

## How to run

Run the full curated pipeline:

```bash
python run_curated_pipeline.py --output-root results/curated_pipeline --overwrite
```

Run a smaller verification pass:

```bash
python run_curated_pipeline.py --output-root results/curated_pipeline --anchor-count 4 --epochs 5 --batch-size 64 --overwrite
```

## Pipeline steps

1. `build_dataset.py` rebuilds the curated HLPE benchmark from the five final checkpoints.
2. `fit_model.py` trains `direct_no_delta_q` and `direct_with_delta_q`.
3. `compare_methods.py` evaluates those models against `delta_q`, `action_l2_distance`, and `mistake_magnitude`.
4. `run_curated_pipeline.py` writes the final markdown summary under `results/.../summary/summary.md`.

## Outputs

The pipeline writes:

- `results/curated_pipeline/dataset`
- `results/curated_pipeline/models`
- `results/curated_pipeline/comparison`
- `results/curated_pipeline/summary/summary.md`
