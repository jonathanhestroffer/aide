# Configuration

AIDE uses configuration to separate **experiment definition** from **framework implementation**.

An experiment should generally be reproducible by changing configuration rather than modifying Python code. Hydra is responsible for composing the configuration, while Pydantic validates the resolved configuration before it enters the training runtime.

This creates a clear boundary:

```text
Configuration Files
        │
        ▼
      Hydra
   Composition
        │
        ▼
Resolved DictConfig
        │
        ▼
    Pydantic
   Validation
        │
        ▼
ExperimentConfig
        │
        ▼
AIDE Runtime
```

This page explains how that system works and how to customize experiments.

---

## Mental Model

An AIDE experiment is composed from small configuration groups rather than one large configuration file.

The experiment file under `configs/experiment/` is the entrypoint. It selects the model, datamodule, trainer, infrastructure, and checkpoint configuration that make up the experiment.

The default scaffolded experiment is:

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

The defaults list is the core of the composition system.

* `/config` provides global Hydra behavior, including run and sweep output directories.
* `/model@trainable.model: cnn` selects the model configuration and places it under `trainable.model`.
* `/datamodule: artifact` configures the artifact-backed datamodule.
* `/trainer: default` defines Lightning runtime settings.
* `/infrastructure: local` configures MLflow and local storage.
* `/checkpoint: default` defines checkpoint behavior.
* `_self_` allows the experiment file itself to override or extend composed values.

The result is one resolved experiment configuration assembled from independently reusable pieces.

---

## Why Configuration Is Structured This Way

AIDE separates concerns that tend to change at different rates.

For example:

```text
Model architecture       → configs/model/
Dataset behavior         → configs/datamodule/
Hardware/runtime         → configs/trainer/
Experiment tracking      → configs/infrastructure/
Checkpoint policy        → configs/checkpoint/
Experiment definition    → configs/experiment/
```

This makes experiment variants cheap to create.

For example, changing from a CPU experiment to a GPU experiment should generally require a configuration change rather than changes to training code.

Likewise, changing learning rate or augmentation should not require modifying the model or datamodule implementation.

The goal is to make the following workflow normal:

```text
Define experiment
       │
       ▼
Select configuration
       │
       ▼
Run experiment
       │
       ▼
Record configuration + results
       │
       ▼
Compare experiments
```

---

## Hydra Composition

Hydra resolves the defaults list into a single configuration.

For example:

```text
configs/
├── config.yaml
├── checkpoint/
│   └── default.yaml
├── datamodule/
│   └── artifact.yaml
├── experiment/
│   └── default.yaml
├── infrastructure/
│   └── local.yaml
├── model/
│   └── cnn.yaml
└── trainer/
    └── default.yaml
```

The experiment does not duplicate the contents of these files. Instead, it selects which configuration from each group should be used.

This is particularly useful when multiple experiments share most of their configuration.

For example:

```text
experiment/default
experiment/gpu
experiment/fast_dev
experiment/no_augmentation
```

can reuse the same model, datamodule, and infrastructure configurations while changing only the pieces relevant to each experiment.

---

## Hydra → Pydantic Boundary

Hydra's responsibility ends after configuration composition.

The resolved Hydra configuration enters `aide.scripts.train` as a `DictConfig`. AIDE then converts that configuration into the typed `ExperimentConfig`.

Conceptually:

```text
Hydra
  │
  │ composition
  ▼
DictConfig
  │
  │ validation / conversion
  ▼
ExperimentConfig
  │
  ├── trainable
  ├── datamodule
  ├── trainer
  ├── checkpoint
  └── infrastructure
  │
  ▼
Runtime construction
```

This distinction is intentional.

**Hydra answers:**

> How should this experiment be composed?

**Pydantic answers:**

> Is the resulting experiment configuration valid?

This means configuration errors can be detected before the training runtime is constructed.

---

## Scaffolded Config Layout

The scaffold creates these configuration groups:

```text
configs/
├── config.yaml
├── experiment/
├── model/
├── datamodule/
├── trainer/
├── infrastructure/
└── checkpoint/
```

Each group has a specific responsibility.

### `configs/config.yaml`

This is the global Hydra configuration.

In the scaffold it primarily controls where Hydra writes run and sweep outputs.

It is **not** the experiment itself.

### `configs/experiment/`

These files define named experiments.

An experiment selects which model, datamodule, trainer, infrastructure, and checkpoint configurations should be composed.

Examples:

```text
default
gpu
fast_dev
ablation_no_aug
```

This is the primary place to define reusable experiment variants.

### `configs/model/`

These files define the registry key and constructor parameters for a model.

Example:

```yaml
class_name: scaffold_cnn
params:
  num_classes: 10
  lr: 1e-3
```

Despite the name `class_name`, AIDE expects a **registry key**, not a fully qualified Python import path.

AIDE resolves the key through `ModelRegistry` and passes `params` to the registered model constructor.

### `configs/datamodule/`

These files define how the datamodule obtains and transforms data.

The scaffold uses an artifact manifest:

```yaml
artifact_uri: ${oc.env:AIDE_DATASET_MANIFEST}
transforms:
  - class_name: scaffold_random_crop
    params:
      size: 32
      padding: 4
      apply_to: [train]
```

Transforms are also registry-backed, allowing project-specific preprocessing and augmentation code to be selected through configuration.

### `configs/trainer/`

These files define Lightning runtime behavior such as:

* `max_epochs`
* `accelerator`
* `devices`
* `precision`

AIDE passes the resulting trainer configuration into the shared Lightning `Trainer` wrapper.

Hardware and runtime concerns therefore remain separate from model implementation.

### `configs/infrastructure/`

These files define operational concerns around the experiment, including:

* MLflow tracking URI
* artifact location
* save directory
* optional plugin modules

This keeps infrastructure configuration separate from model and dataset configuration.

### `configs/checkpoint/`

These files define checkpoint policy, including:

* save cadence
* monitored metric
* checkpoint naming
* retention behavior

The resulting configuration is consumed by AIDE's callback factory.

---

## Configuration and the Plugin System

Configuration selects **which implementation** should be used, while the plugin system provides the implementation.

For example:

```yaml
class_name: my_model
params:
  lr: 3e-4
```

The configuration does not contain a Python import path.

Instead:

```text
class_name
    │
    ▼
ModelRegistry
    │
    ▼
Registered model
    │
    ▼
params
    │
    ▼
Model instance
```

This allows the same configuration mechanism to work with models supplied by the scaffolded project.

The same pattern is used for components and transforms.

See [Plugins and Registries](PluginsAndRegistries.md) for the implementation details.

---

## Common Customization Patterns

### Create a New Experiment Variant

Create a new file under `configs/experiment/`.

For example:

`configs/experiment/gpu.yaml`

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

The model and datamodule remain unchanged while the trainer configuration changes.

This is preferable to duplicating the entire experiment configuration.

### Swap the Model

Create a new model configuration:

```text
configs/model/
├── cnn.yaml
└── my_model.yaml
```

Then reference it from an experiment:

```yaml
- /model@trainable.model: my_model
```

The training runtime does not need to change.

### Change Dataset Transforms

Modify the `transforms` list in the datamodule configuration.

This is the appropriate place for configuration-driven changes such as:

* augmentation
* normalization
* train/validation behavior
* transform parameters

The transform implementation itself belongs in the project's plugin package.

---

## Command-Line Overrides

The generated `train.py` wrapper intentionally exposes a simple interface:

```bash
aide train --experiment default
```

It primarily handles project discovery and experiment selection.

When arbitrary Hydra overrides are required, use the lower-level training entrypoint.

For example:

```bash
set -a
source .env
set +a

python -m aide.scripts.train \
    --config-path ./configs \
    --config-name experiment/default \
    trainer.max_epochs=5
```

A model parameter can similarly be overridden:

```bash
python -m aide.scripts.train \
    --config-path ./configs \
    --config-name experiment/default \
    trainable.model.params.lr=3e-4
```

This distinction is intentional:

```text
Named / reusable change
        │
        ▼
Configuration file

Temporary experiment change
        │
        ▼
Hydra CLI override
```

If a setting becomes part of a repeatable experiment, it should generally graduate from a command-line override into a named configuration.

---

## Reproducibility

Configuration is also part of experiment provenance.

A training result should not be considered reproducible merely because the model checkpoint exists.

The experiment configuration determines:

* which model was trained
* which model parameters were used
* which dataset and transforms were used
* which trainer settings were used
* which hardware/runtime configuration was selected
* which checkpoint policy was active
* where artifacts were stored

AIDE therefore persists the resolved experiment configuration alongside experiment artifacts.

Conceptually:

```text
                    Experiment
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Config         Metrics      Checkpoint
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                     MLflow
```

This makes configuration a first-class part of the experiment record rather than merely a mechanism for launching training.

---

## Practical Guidance

Use the configuration hierarchy deliberately:

| Concern                    | Configuration location    |
| -------------------------- | ------------------------- |
| Experiment definition      | `configs/experiment/`     |
| Model selection/parameters | `configs/model/`          |
| Dataset/transforms         | `configs/datamodule/`     |
| Hardware/training runtime  | `configs/trainer/`        |
| Tracking/storage           | `configs/infrastructure/` |
| Checkpoint policy          | `configs/checkpoint/`     |
| Global Hydra behavior      | `configs/config.yaml`     |

Prefer:

* composition over duplication
* named experiment variants over undocumented CLI commands
* configuration for behavior that should be reproducible
* CLI overrides for temporary experimentation
* plugin code for new implementations
* typed validation before entering the training runtime

The resulting separation is:

```text
Configuration
    │
    │ selects behavior
    ▼
AIDE Runtime
    │
    │ provides infrastructure
    ▼
Project Plugins
    │
    │ provide implementations
    ▼
Model / Data / Components
```

That separation is one of the central architectural goals of AIDE.

## Related Pages

* [Home](Home.md)
* [TrainableModel](TrainableModel.md)
* [Trainer and Checkpoints](TrainerAndCheckpoints.md)
* [Plugins and Registries](PluginsAndRegistries.md)
