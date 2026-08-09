# AIDE

**AIDE** is the **AI Development & Engineering Framework**.

It is a reusable training framework for teams or individual users who want a consistent way to
define experiments, register custom models and components, run training through Lightning, and
track results with MLflow without rebuilding the same project boilerplate every time.

## What AIDE Handles

- Project scaffolding for new experiments
- Hydra config composition
- Pydantic validation of resolved experiment config
- Registry-based loading of models, components, and transforms
- A unified Lightning trainer wrapper
- MLflow logging for metrics, config artifacts, and checkpoints
- Shared dataset artifact locations for scaffolded projects

## Quick Start

Install AIDE from the repository root:

```bash
uv sync --frozen
source .venv/bin/activate
```

Create a scaffolded experiment project:

```bash
aide-init example
cd example
```

This creates a project with:

```text
example/
  .env
  configs/
  plugins/
  train.py
```

By default, `aide-init` uses the current working directory as the shared artifact location and
points `AIDE_DATASET_MANIFEST` at:

```text
./cifar10/manifest.json
```

If you want multiple projects to share a different dataset location, pass `--artifact-dir`:

```text
/shared/aide/artifacts/cifar10/manifest.json
```

```bash
aide-init example --artifact-dir /shared/aide/artifacts
```

Run the default experiment:

```bash
python train.py --experiment=default
```

That launcher reads `.env`, resolves the project config path, composes the selected Hydra
experiment, loads local plugins, validates the result into `ExperimentConfig`, and starts the
shared Lightning training runtime.

## How Configuration Works

AIDE uses Hydra config composition, not one large hand-written config file.

The scaffold gives you config groups like these:

- `configs/config.yaml`: global Hydra behavior, including run and sweep output directories
- `configs/experiment/default.yaml`: the experiment entrypoint that composes the final run config
- `configs/model/`: model selections and parameters
- `configs/datamodule/`: dataset manifest and transform chain
- `configs/trainer/`: Lightning trainer settings such as epochs, accelerator, devices, precision
- `configs/infrastructure/`: MLflow tracking URI, artifact location, save directory
- `configs/checkpoint/`: checkpoint policy for save cadence, monitored metric, and file naming

The default experiment is implemented as a Hydra defaults list:

```yaml
defaults:
  - /config
  - /model@trainable.model: cnn
  - /datamodule: artifact
  - /trainer: default
  - /infrastructure: local
  - /checkpoint: default
  - _self_
```

That means AIDE composes a final config from several smaller files, then `aide.scripts.train`
converts the resolved Hydra config into a typed `ExperimentConfig`. From there:

- `trainable.model` is built from `ModelRegistry`
- `datamodule` is built from the artifact manifest path in config
- `trainer` fields are passed into the shared Lightning `Trainer`
- `checkpoint` controls checkpoint cadence, naming, and monitored metric
- `infrastructure` controls MLflow tracking, artifact storage, and output locations

The practical workflow is simple: change config to select behavior, and only write Python when
you need a new model, transform, or component.

For full Hydra overrides beyond selecting `--experiment`, use the lower-level training entrypoint
instead of the generated `train.py` wrapper. The wrapper only parses `--experiment`.

## Add a Model or Custom Component

Custom code lives in the generated project's `plugins/` package.

To add a model:

1. Create a class that inherits from `TrainableModel`.
2. Register it with `ModelRegistry`.
3. Point config at the registered key.

```python
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(TrainableModel): ...
```

Then select it in Hydra config by setting the model config's `class_name` to `my_model`.

Custom components follow the same pattern:

- `ComponentRegistry` for preprocessors and postprocessors used by a model
- `TransformRegistry` for dataset transforms applied by the datamodule

At startup, AIDE imports the local plugin package before construction so registration side
effects populate the registries. The factories then instantiate the configured classes from
their `class_name` and `params` fields.

## Outputs

By default, a scaffolded project writes to `workspace/`:

- `workspace/hydra_logs/` for Hydra run metadata
- `workspace/lightning_logs/` for Lightning outputs
- `workspace/metadata/aide.db` for local MLflow tracking
- `workspace/artifacts/` for MLflow artifacts such as config snapshots and checkpoints

Override these with environment variables such as `AIDE_TRACKING_URI`,
`AIDE_ARTIFACT_LOCATION`, and `AIDE_SAVE_DIR`.

## Commands

```bash
aide-init <directory> --artifact-dir <dir>
aide init <experiment-name>
aide train [Hydra arguments]
aide list <project-path> [--kind models|components|transforms]
```

For scaffolded projects, `python train.py --experiment=...` is the preferred entrypoint because
it automatically uses the project's `.env` and `configs/` directory.

If you need raw Hydra overrides, source the project environment first and then run the lower-level
entrypoint with an explicit config path, for example:

```bash
set -a
source .env
set +a
python -m aide.scripts.train --config-path ./configs --config-name experiment/default trainer.max_epochs=5
```

## Scope

The current scaffold is centered on a local artifact-backed datamodule and a CIFAR-10 example.
The README stays focused on getting started and understanding the core concepts.

## Further Reading

For deeper implementation details, see the wiki pages in `wikis/`:

- `wikis/Home.md`
- `wikis/Configuration.md`
- `wikis/TrainableModel.md`
- `wikis/TrainerAndCheckpoints.md`
- `wikis/PluginsAndRegistries.md`
