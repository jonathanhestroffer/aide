# Configuration

This page explains how AIDE uses Hydra configuration composition and how to customize
experiments without turning every change into a Python change.

## Mental Model

In AIDE, an experiment is composed from small configuration groups rather than one large file.
The experiment file under `configs/experiment/` is the entrypoint. It pulls in the global
Hydra config, a model selection, a datamodule config, trainer settings, infrastructure settings,
and checkpoint settings.

There are two common ways to launch this composed config:

- `python train.py --experiment=default` for the scaffolded project workflow
- `python -m aide.scripts.train --config-path ./configs --config-name experiment/default ...`
	when you need direct Hydra arguments

The default scaffolded experiment looks like this:

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

That defaults list is the core of the system.

- `/config` brings in global Hydra behavior, especially run and sweep output directories.
- `/model@trainable.model: cnn` injects the chosen model config into `trainable.model`.
- `/datamodule: artifact` configures dataset loading from a manifest.
- `/trainer: default` controls Lightning runtime settings.
- `/infrastructure: local` configures MLflow and local output paths.
- `/checkpoint: default` provides the top-level checkpoint policy for the run.
- `_self_` lets the current experiment file override or extend the composed values.

## How It Maps to Runtime

Hydra resolves the composed config and passes it into `aide.scripts.train` as a `DictConfig`.
AIDE then converts that resolved config into a typed `ExperimentConfig` using Pydantic.

From there:

- `trainable` becomes the model pipeline configuration.
- `datamodule` becomes the artifact-backed Lightning datamodule.
- `trainer` is passed into the shared Lightning `Trainer` wrapper.
- `checkpoint` becomes the top-level checkpoint configuration consumed by the callback factory.
- `infrastructure` controls MLflow tracking, artifact storage, and save directories.

This gives you two layers of safety:

1. Hydra composes the experiment from reusable groups.
2. Pydantic validates the final shape before training begins.

## Scaffolded Config Layout

The scaffold creates these config groups:

- `configs/config.yaml`
- `configs/experiment/`
- `configs/model/`
- `configs/datamodule/`
- `configs/trainer/`
- `configs/infrastructure/`
- `configs/checkpoint/`

Each group has a clear role.

### `configs/config.yaml`

This is the global Hydra config. In the scaffold it mainly sets where Hydra writes run and sweep
outputs. It is not the experiment itself.

### `configs/experiment/`

These files define named experiments. An experiment chooses which model, datamodule, trainer,
infrastructure, and checkpoint configs to compose together.

This is where you should define variants like:

- `default`
- `gpu`
- `fast_dev`
- `ablation_no_aug`

### `configs/model/`

These files define the registry key and constructor parameters for the selected model.

Typical contents:

```yaml
class_name: scaffold_cnn
params:
	num_classes: 10
	lr: 1e-3
```

Despite the field name `class_name`, AIDE expects a registry key such as `scaffold_cnn`, not a
fully qualified Python import path. That key is looked up in `ModelRegistry`, and `params` are
passed to the model class constructor.

### `configs/datamodule/`

These files define how data is loaded.

In the scaffold, the datamodule config points at `AIDE_DATASET_MANIFEST` and optionally defines
a transform chain. Each transform is a registry-backed component with a registry key in
`class_name` and its constructor values in `params`.

Typical contents:

```yaml
artifact_uri: ${oc.env:AIDE_DATASET_MANIFEST}
transforms:
	- class_name: scaffold_random_crop
		params:
			size: 32
			padding: 4
			apply_to: [train]
```

### `configs/trainer/`

These files define Lightning runtime behavior such as:

- `max_epochs`
- `accelerator`
- `devices`
- `precision`

AIDE passes these values directly into `lightning.Trainer`.

### `configs/infrastructure/`

These files define operational settings around training rather than the model itself, including:

- MLflow tracking URI
- artifact location
- save directory
- optional plugin modules listed in config

### `configs/checkpoint/`

These files define checkpoint behavior. AIDE merges this config into
the top-level `checkpoint` section, then uses it to build the checkpoint callback.

## Common Customization Patterns

### Create a New Experiment Variant

If you want a new experiment variant, create a new file under `configs/experiment/` and reuse the
existing groups.

Example: `configs/experiment/gpu.yaml`

```yaml
defaults:
	- /config
	- /model@trainable.model: cnn
	- /datamodule: artifact
	- /trainer: gpu
	- /infrastructure: local
	- /checkpoint: default
	- _self_
```

This lets you swap trainer behavior without changing model or datamodule logic.

### Swap the Model

To try a new registered model, add a new config file under `configs/model/` and point the
experiment at it.

### Change Dataset Transforms

Update the datamodule config's `transforms` list. This is usually the right place to add
augmentation, normalization, or split-specific transform behavior.

### Override at the Command Line

The generated `train.py` wrapper only supports selecting `--experiment`. It does not forward
arbitrary Hydra arguments.

For direct Hydra overrides, source the scaffold's `.env` file and call the lower-level training
entrypoint yourself:

```bash
set -a
source .env
set +a
python -m aide.scripts.train --config-path ./configs --config-name experiment/default trainer.max_epochs=5
```

or:

```bash
set -a
source .env
set +a
python -m aide.scripts.train --config-path ./configs --config-name experiment/default trainable.model.params.lr=3e-4
```

Use config files when a setting should be named and reusable. Use CLI overrides when the change
is temporary.

## Practical Guidance

- Put stable run shapes in named experiment files.
- Put reusable model parameter sets in `configs/model/`.
- Put data loading and augmentation behavior in `configs/datamodule/`.
- Put hardware and runtime behavior in `configs/trainer/`.
- Put checkpoint policy in `configs/checkpoint/`.
- Put tracking and storage concerns in `configs/infrastructure/`.
- Prefer composition over copying large config files.

## Related Pages

- `wikis/TrainableModel.md`
- `wikis/TrainerAndCheckpoints.md`
- `wikis/PluginsAndRegistries.md`
