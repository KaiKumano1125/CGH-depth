# Project Structure

The current notebook experiments can be moved toward a reusable layout:

```text
configs/
  base.toml
  experiments/
    only_frequency.toml
scripts/
  train_experiment.py
  run_inference.py
src/
  cgh_depth/
    config.py
    datasets.py
    encoders.py
    models.py
    training.py
    inference.py
```

## What moved out of the notebook

- Dataset loading
- Frequency encoder logic
- U-Net model definition
- Training loop
- Validation loop
- Checkpoint loading
- Single-image inference
- Batch inference

## Typical workflow

Use the project virtual environment so `torch`, `tensorboard`, and `pyexr` match the notebook environment:

```powershell
.\.venv\Scripts\python.exe
```

Inspect the dataset wiring:

```powershell
.\.venv\Scripts\python.exe scripts/train_experiment.py --inspect-only
```

Run training:

```powershell
.\.venv\Scripts\python.exe scripts/train_experiment.py --config configs/experiments/only_frequency.toml
```

Run one test prediction:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml
```

Run batch inference:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --config configs/experiments/only_frequency.toml --batch
```

## Notebook usage

You can now keep notebooks focused on analysis and visualization:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd().parents[1] / "src"))

from cgh_depth.config import load_experiment_config
from cgh_depth.training import inspect_dataset

cfg = load_experiment_config("configs/experiments/only_frequency.toml")
inspect_dataset(cfg)
```
