# ML Platform

Personal project to demonstrate practical ML engineering best practices and provide a reusable framework for model development and deployment.

## Project Goal

This repository is designed to make experimentation and productionization feel like the same workflow. The core idea is:

1. Define task-specific components (models, losses, optimizers, trainers).
2. Register components by key.
3. Build them from strongly typed configuration.
4. Train and evaluate through a generic trainer interface.
5. Track artifacts and metrics consistently.
6. Run Hydra-style experiments with clean configuration overrides.

The framework handles orchestration, while end users focus on domain logic.

## Design Principles

- Registry-first extensibility: New components are discovered and selected by config key, not hardcoded branches.
- Strongly typed config contracts: Pydantic-based configs provide validation and safer refactors.
- Separation of concerns: Model code, training loops, and deployment targets stay decoupled.
- Reproducibility by default: Config + artifact + metric outputs should fully describe an experiment.
- Deployability mindset: The same project includes local training, API serving, and infra/deployment scaffolding.

## Developer Experience

### 1) Add a custom component

Implement your class in the relevant package, then register it with a key.

```python
from aide.registry.map import ModelRegistry


@ModelRegistry.register("resnet18")
class ResNet18Model(...): ...
```

The same pattern applies for losses, optimizers, and trainers.

### 2) Configure via typed config

Define or extend the relevant config schema so runtime options are validated and documented.

### 3) Select by key at runtime

Choose your component in config, then let the generic build path instantiate the selected class.

### 4) Train with shared interfaces

Use a generic trainer API to keep loops, logging, and lifecycle handling consistent across tasks.

### 5) Track metrics and artifacts

Persist checkpoints, metrics, and metadata using a standardized tracking interface so runs are comparable.

### 6) Run Hydra-style experiments

Use config composition and overrides to launch controlled experiment sweeps without changing source code.

## Repository Layout

```text
aide/
├── api/                    # Inference API entrypoint(s)
├── configs/                # Base and environment-specific configuration
├── docker/                 # Dockerfiles for training and serving
├── kubernetes/             # Deployment and service manifests
├── src/aide/
│   ├── core/               # Core framework abstractions (e.g., Registry)
│   ├── models/             # User-defined model implementations
│   ├── losses/             # User-defined loss implementations
│   ├── optimizers/         # User-defined optimizer implementations
│   ├── trainers/           # Generic and task-specific trainers
│   ├── registry/           # Registry instances and discovery hooks
│   ├── config.py           # Strongly typed config definitions
│   ├── train.py            # Training entrypoint
│   └── inference.py        # Inference utilities
├── terraform/              # Cloud infrastructure scaffolding
└── tests/                  # Unit, integration, and load tests
```

## Why This Exists

Most ML projects start simple and then accumulate ad hoc wiring as they scale. This repo is a deliberate attempt to keep the architecture clean from day one:

- simple for new contributors,
- strict enough for reliability,
- flexible enough for new tasks and models,
- and close to production concerns.

## Current Status

This is an evolving framework. The intended end state is a robust baseline where a developer can add a new task by implementing and registering components, then run full training and deployment workflows through configuration.

## Running Experiments With Hydra

Users can install this package and run training from their own project configs.

### Built-In CIFAR-10 Transforms

The package now includes useful transform components for CIFAR-10 artifact datasets:

- `cifar10_random_crop` (defaults to train split, 32x32 crop with padding)
- `cifar10_random_horizontal_flip` (defaults to train split)
- `cifar10_normalize` (defaults to all splits with CIFAR-10 mean/std)

Example config snippet:

```yaml
datamodule:
	transforms:
		- class_name: cifar10_random_crop
			params:
				crop_size: 32
				padding: 4
				apply_to: [train]
		- class_name: cifar10_random_horizontal_flip
			params:
				flip_probability: 0.5
				apply_to: [train]
		- class_name: cifar10_normalize
			params:
				apply_to: [all]
```

### Example User Project

The CIFAR-10 example model and transforms now live under [user_project](user_project),
separate from the framework package and outside the installable `aide` distribution.
That package has its own Hydra config at [user_project/configs/train.yaml](user_project/configs/train.yaml) and can be run
directly with:

```bash
python -m user_project.train
```

That keeps `aide` focused on the framework/runtime layer while `user_project`
simulates custom user-owned code.

### Pip-Installed Quick Start

`aide` runs the packaged module entrypoint (`python -m aide.train`) and
defaults to the packaged config at `aide/configs/train.yaml`.

CLI supports subcommands:

```bash
aide train --config-path ./configs --config-name train
```

1. Install package:

```bash
pip install aide
```

2. Provide dataset manifest path (required unless you have `./artifacts/datasets.json`):

```bash
export AIDE_DATASET_MANIFEST=/abs/path/to/datasets.json
```

3. Run training with packaged defaults:

```bash
aide
```

Optional environment overrides for outputs/tracking:

```bash
export AIDE_SAVE_DIR=/abs/path/to/workspace
export AIDE_TRACKING_URI=sqlite:////abs/path/to/workspace/aide.db
export AIDE_ARTIFACT_LOCATION=file:///abs/path/to/workspace/artifacts
```

If users want full control, they can still provide their own Hydra config folder:

```bash
aide --config-path ./configs --config-name train
```

### Single Experiment (Monolithic Config)

```bash
aide --config-path ./configs --config-name train
```

### Sweep / Multirun

```bash
aide -m --config-path ./configs --config-name train \
	trainable.model.params.lr=1e-3,3e-4,1e-4 \
	trainer.max_epochs=5,10
```

### Loading User Plugins

Custom models/components can live in a separate user project. Register classes with decorators, then list plugin modules in config:

```yaml
infrastructure:
	plugins:
		- my_project.models
		- my_project.components
```

At runtime, training imports these modules from config so registration side effects populate registries before model/datamodule construction.
