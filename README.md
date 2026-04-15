# Curated RoboPianist Pipeline

This repo now keeps one path only: build the curated HLPE dataset, train the two direct intervention models, compare them against `delta_q`, and write a short summary.

## Setup

Use Python 3.10 and install the small dependency set:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline expects the five curated checkpoints to live under `robopianist_rl_runs/` in the paths hardcoded in `run_curated_pipeline.py`.

## Run

One command runs the whole thing:

```bash
python run_curated_pipeline.py --output-root results/curated_pipeline --overwrite
```

Useful knobs if you want a smaller smoke run first:

```bash
python run_curated_pipeline.py --anchor-count 4 --epochs 3 --batch-size 64 --overwrite
```

## Outputs

The run writes:

- `results/curated_pipeline/dataset`
- `results/curated_pipeline/models`
- `results/curated_pipeline/comparison`
- `results/curated_pipeline/summary/summary.md`
# robopianist-cost-aware-intervention
