# Predictive Maintenance: Remaining Useful Life (RUL) for Turbofan Engines

CS5998 Capstone Project — C.N.D. Anthony (268449F)

An end-to-end predictive maintenance system that estimates the Remaining Useful
Life of turbofan engines from multivariate sensor telemetry (NASA CMAPSS), and
raises maintenance alerts when predicted RUL drops below set thresholds.

## Scope

| Item | Choice |
|------|--------|
| Problem | Time-series regression of cycle-level RUL (piecewise-linear target, cap = 125) |
| Data | NASA CMAPSS — FD001 (primary), FD002 (six operating conditions) |
| Baselines | Random Forest, XGBoost on 30-cycle rolling-window features |
| Advanced model | Hybrid 1D-CNN + BiLSTM on raw sensor sequences |
| Metrics | RMSE, NASA asymmetric scoring function |
| Serving | Dockerised FastAPI scoring service, Streamlit fleet monitoring UI |

See [docs/](docs/) for the Project Definition Document.

## Repository layout

```
configs/      Experiment and pipeline configuration (YAML)
data/         Raw and processed data (git-ignored, created by scripts)
docs/         Project documents and diagrams
models/       Trained model artefacts (git-ignored)
notebooks/    Exploratory analysis notebooks
src/          Source code
  data/       Data acquisition and loading
tests/        Unit tests
```

## Getting started

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Get the data

Download CMAPSS from the NASA Prognostics Data Repository and extract it to
`data/raw/CMAPSSData/`:

```bash
python -m src.data.download
```

If you already have `CMAPSSData.zip` locally, extract it without downloading:

```bash
python -m src.data.download --zip path/to/CMAPSSData.zip
```

### Load a subset

```python
from src.data.load import load_subset

train, test, rul = load_subset("FD001")
```

### Run tests

```bash
pytest
```

## Milestones

| Milestone | Week | Deliverables | Status |
|-----------|------|--------------|--------|
| M1 | 4 | Project definition, repository, data acquisition pipeline | In progress |
| M2 | 8 | EDA, sliding-window pipeline, XGBoost + 1D-CNN/BiLSTM models, architecture diagram, demo | Planned |
| M3 | 12 | Final report, reproducibility checklist, Dockerised FastAPI service, Streamlit UI | Planned |

## Data source

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation," *International Conference on
Prognostics and Health Management (PHM08)*, 2008. NASA Ames Prognostics Data
Repository.
