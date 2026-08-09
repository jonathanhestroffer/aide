# AIDE

**AIDE** is the **AI Development & Engineering Framework**: a reusable foundation for
building, training, and tracking machine-learning experiments without recreating the same
project plumbing for every model or user.

AIDE standardizes the parts of an ML project that tend to become repetitive and fragile:
experiment configuration, component discovery, training hardware settings, checkpoints,
run outputs, artifact logging, and experiment tracking. Model authors keep ownership of their
models and data code; AIDE supplies the shared runtime around them.

## What AIDE Does

- Creates self-contained experiment projects with configuration, plugins, scripts, and an
  environment file.
- Uses Pydantic-validated Hydra configuration to compose an experiment from model, data,
  trainer, infrastructure, and checkpoint settings.
- Discovers models, components, and dataset transforms through decorator-based registries.
- Runs any model that implements AIDE's `TrainableModel` interface through a single
  Lightning `Trainer` wrapper.
- Delegates accelerator, device count, precision, and training duration to Lightning's
  configurable trainer settings.
- Tracks Lightning metrics with MLflow, stores the resolved experiment configuration as an
  MLflow artifact, and uploads saved checkpoints to artifact storage.
- Loads artifact-backed train, validation, and optional test datasets from a JSON manifest.

## Who It Is For

AIDE is designed for teams or individual practitioners who want a consistent training runtime
across many models and projects. Each user can create an independent scaffold with its own
`configs/`, `plugins/`, `.env`, datasets, and workspace outputs, while reusing the same AIDE
installation and conventions.

## Quick Start

The default workflow creates an experiment named `example`, creates or reuses CIFAR-10
artifacts, and trains the scaffolded CNN with the default configuration.

### 1. Install AIDE with uv

From the AIDE repository root, create the locked environment and install the project:

```bash
uv sync --frozen
source .venv/bin/activate
```

The activation step makes the installed `aide` and `aide-init` launchers available in your
shell. `uv` is used for environment and dependency installation; run generated project scripts
with Python normally.

### 2. Create an Experiment Project

```bash
aide-init example --artifact-dir /shared/aide/artifacts
cd example
```

`aide-init` generates a project directory containing:

```text
example/
  .env                         # Project, config, plugin, and data locations
  configs/                     # Hydra configuration groups
  plugins/                     # User-owned models and transforms
  train.py                     # Project training launcher
```

The `--artifact-dir` argument specifies where AIDE stores the shared dataset. AIDE creates or
reuses `/shared/aide/artifacts/cifar10/manifest.json` and the associated `train.pt`, `val.pt`,
and `test.pt` files in the same directory. The generated `.env` sets `AIDE_DATASET_MANIFEST`
to the resolved manifest path, so multiple projects can reuse one CIFAR-10 download.

Omit `--artifact-dir` to create project-local artifacts at `example/data/cifar10/`.

### 3. Run the Default Experiment

```bash
python train.py --experiment=default
```

The launcher reads the project `.env`, composes `configs/experiment/default.yaml`, imports the
local plugins, validates the resolved configuration, and starts the unified Lightning trainer.

## Experiment Outputs

The local scaffold stores runtime outputs under `workspace/` by default:

- `workspace/hydra_logs/`: Hydra run configuration and launch metadata.
- `workspace/lightning_logs/`: Lightning logs and local checkpoint output.
- `workspace/metadata/aide.db`: local SQLite MLflow tracking database.
- `workspace/artifacts/`: MLflow artifacts, including `experiment_config.json` and uploaded
  checkpoints.

Set these environment variables before training to use different locations or a remote MLflow
tracking server:

```bash
export AIDE_TRACKING_URI=http://mlflow.example.com
export AIDE_ARTIFACT_LOCATION=s3://my-mlflow-artifacts
export AIDE_SAVE_DIR=/path/to/lightning-logs
```

## Configuration and Hardware

Experiments are composed from configuration groups under `configs/experiment/`. The default
experiment selects a model, artifact datamodule, trainer, local infrastructure, and checkpoint
policy. Change the trainer configuration to select Lightning-supported hardware behavior:

```yaml
# configs/trainer/default.yaml
max_epochs: 100
accelerator: auto
devices: auto
precision: "32"
```

For example, set `accelerator: gpu`, an appropriate `devices` value, and supported precision
for a GPU workload. AIDE passes these fields directly to `lightning.Trainer`.

## Add a Model or Transform

User-owned code belongs in the generated project's `plugins/` directory. Register a Lightning
model that inherits from `TrainableModel`, then select it by key in configuration:

```python
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(TrainableModel): ...
```

The framework imports project plugin files before it builds the configured model and datamodule.
Models must implement Lightning's `forward`, `training_step`, and `configure_optimizers`
methods. Optional preprocessor and postprocessor components are also selected by configuration.

## AIDE Commands

```bash
aide-init <directory> --artifact-dir <dir> # Create a scaffold and configure shared CIFAR-10 artifacts
aide init <experiment-name>              # Create <experiment-name> under the current directory
aide train [Hydra arguments]             # Start training through the AIDE CLI
aide list <project-path> [--kind models] # Inspect registered models, components, or transforms
```

The generated `train.py` is the recommended launcher for a scaffolded project because it reads
the local `.env` file and selects the project configuration path automatically.

## Repository Layout

```text
src/aide/
  core/          # Typed configuration, datamodule, trainer, logging, and base abstractions
  callbacks/     # Checkpoint callback and artifact upload behavior
  registry/      # Registries and plugin discovery
  scaffold/      # Templates and project generator
  scripts/       # CLI entrypoints: aide, aide init, and aide list
example/         # Checked-in scaffold example
pyproject.toml   # Dependencies, uv configuration, and console scripts
```

## Current Scope

AIDE currently provides a local artifact datamodule for PyTorch `Dataset` objects stored in a
JSON split manifest, plus a CIFAR-10 scaffold example. The framework is extensible through
plugins, but remote dataset URI handling and production deployment backends are not implemented
by the current runtime.
