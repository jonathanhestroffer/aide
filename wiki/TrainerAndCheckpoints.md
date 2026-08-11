# Trainer and Checkpoints

This page explains how AIDE wraps Lightning's trainer, integrates MLflow, and manages
checkpoint persistence.

The trainer layer is intentionally thin. AIDE does not attempt to replace Lightning's
training semantics. Instead, it provides a consistent orchestration layer around training,
configuration, logging, callbacks, and artifact persistence.

## Trainer Architecture

AIDE separates experiment configuration from training execution and infrastructure concerns.

The high-level flow is:

```text
                    ExperimentConfig
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        Trainable      DataModule      Trainer
             │                           │
             │                           │
             └──────────────┬────────────┘
                            ▼
                     Lightning Trainer
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
               MLflow             Checkpoints
                  │                   │
                  └─────────┬─────────┘
                            ▼
                       Artifacts
```

This separation is intentional:

* **Model configuration** determines what is trained.
* **Datamodule configuration** determines how data is provided.
* **Trainer configuration** determines how training executes.
* **Checkpoint configuration** determines how model state is persisted.
* **Infrastructure configuration** determines where tracking and artifacts are stored.
* **MLflow** provides experiment tracking and artifact persistence.

The result is a training runtime that can be reused across different experiments without
requiring each project to recreate the same training infrastructure.

## Trainer Overview

AIDE uses a small wrapper around `lightning.Trainer`.

Its responsibilities are to:

* construct the MLflow-backed logger
* construct configured callbacks
* translate validated trainer configuration into Lightning arguments
* coordinate logging of the resolved experiment configuration
* invoke the Lightning training lifecycle

The wrapper is intentionally small. The goal is to centralize infrastructure and
orchestration boilerplate without replacing Lightning's core behavior.

AIDE therefore remains compatible with the standard Lightning programming model.

If a user knows how to configure and use `lightning.Trainer`, the AIDE trainer should
behave predictably.

## Trainer Configuration

The scaffolded trainer configuration controls a subset of Lightning trainer options directly:

* `max_epochs`
* `accelerator`
* `devices`
* `precision`

These values come from `configs/trainer/` and are forwarded into `lightning.Trainer`.

For example:

```yaml
max_epochs: 100
accelerator: auto
devices: auto
precision: "32"
```

This keeps hardware and runtime behavior in configuration rather than embedding those
decisions in model code or handwritten launch scripts.

Additional trainer configurations can be created for common execution profiles such as:

* CPU debugging
* single-GPU training
* mixed precision
* short development runs
* CI smoke tests

For example:

```text
configs/trainer/
├── default.yaml
├── cpu_debug.yaml
├── gpu.yaml
└── smoke_test.yaml
```

A named trainer configuration can then be selected by an experiment configuration.

## Why Trainer Configuration Is Separate

AIDE deliberately separates model configuration from execution configuration.

A model should not need to know whether it is running:

* on CPU
* on one GPU
* with mixed precision
* for 5 epochs
* for 100 epochs
* locally
* in CI

Those are execution concerns.

For example, the same model configuration can be used with different trainer profiles:

```text
                    Model
                      │
             ┌────────┴────────┐
             │                 │
        CPU Debug          GPU Training
             │                 │
       trainer.cpu       trainer.gpu
```

This allows experiments to change execution characteristics without modifying model code.

## MLflow Integration

AIDE creates an `MLFlowLoggerAdapter` and attaches it to Lightning.

The adapter provides two capabilities:

1. It behaves as a normal Lightning logger for training metrics.
2. It exposes direct MLflow client functionality for artifact-oriented operations.

This allows the training runtime to use Lightning's normal logging interface while still
having access to MLflow operations when infrastructure-level artifact handling is required.

For example, training code can continue to use Lightning's standard:

```python
self.log("train_loss", loss)
```

while infrastructure code can use MLflow operations for artifacts such as configuration
snapshots and checkpoints.

## Resolved Configuration Logging

Before training begins, AIDE logs the resolved `ExperimentConfig` as:

```text
experiment_config.json
```

This is important because the configuration supplied by the user is not necessarily the
same as the final configuration used by the training process.

Hydra may compose values from multiple configuration groups:

```text
experiment/default.yaml
        │
        ├── model/cnn.yaml
        ├── datamodule/artifact.yaml
        ├── trainer/default.yaml
        ├── infrastructure/local.yaml
        └── checkpoint/default.yaml
                │
                ▼
        Resolved ExperimentConfig
                │
                ▼
        experiment_config.json
```

The resolved configuration therefore provides a snapshot of the actual experiment
configuration used for the run.

## Output and Tracking Configuration

Infrastructure settings live under the `infrastructure` configuration group.

Important values include:

* `tracking_uri`
* `artifact_location`
* `save_dir`

The distinction is intentional:

| Configuration    | Responsibility                          |
| ---------------- | --------------------------------------- |
| `trainer`        | How training executes                   |
| `checkpoint`     | How model state is saved                |
| `infrastructure` | Where tracking and artifacts are stored |
| `model`          | What model is trained                   |
| `datamodule`     | How data is provided                    |

This keeps operational concerns separate from model and experiment definitions.

In the scaffolded project, these values default to paths under the project workspace and
can be overridden through configuration or environment variables.

## Checkpoint Flow

AIDE uses Lightning's `ModelCheckpoint` mechanism through its
`ArtifactModelCheckpoint` callback.

The checkpoint lifecycle has two stages:

```text
Lightning training
       │
       ▼
ModelCheckpoint
       │
       ▼
Local checkpoint directory
       │
       ▼
MLflow artifact storage
```

First, Lightning writes checkpoint files to the configured local directory.

After training, AIDE synchronizes the checkpoint directory with MLflow artifact storage.

This separates two responsibilities:

1. **Checkpoint generation** — handled by Lightning.
2. **Artifact persistence** — handled by AIDE's MLflow integration.

AIDE therefore does not reimplement checkpoint serialization. It extends Lightning's
existing checkpoint lifecycle with artifact persistence.

## What AIDE's Checkpoint Callback Adds

`ArtifactModelCheckpoint` subclasses Lightning's `ModelCheckpoint`.

Lightning already provides:

* monitored metrics
* best-checkpoint selection
* `save_top_k`
* `save_last`
* epoch-based checkpointing
* step-based checkpointing
* checkpoint naming
* local checkpoint storage

AIDE adds artifact synchronization so that checkpoints become part of the tracked MLflow
run.

This means the checkpoint is not only a local file produced during training. It becomes a
run artifact that can be associated with the experiment's metrics and configuration.

## Artifact Persistence

The distinction between local checkpointing and artifact persistence is important.

During training:

```text
Training Process
      │
      ▼
Local Filesystem
      │
      ├── checkpoint files
      ├── Hydra outputs
      └── training logs
```

After training:

```text
Local Filesystem
      │
      ▼
MLflow Artifact Store
      │
      ├── experiment_config.json
      └── checkpoints/
```

This allows the local filesystem to act as the working directory while MLflow provides
persistent experiment-level storage.

If artifact upload fails, AIDE intentionally does not fail the training cleanup process.

Artifact synchronization is treated as a post-training persistence operation rather than
part of the optimization loop itself.

## How Checkpoint Paths Are Chosen

If `checkpoint.dirpath` is explicitly configured, AIDE uses that path.

If it is not configured:

* when Hydra is initialized, AIDE uses Hydra's runtime output directory and appends
  `checkpoints/`
* otherwise, AIDE falls back to `infrastructure.save_dir/checkpoints`

This keeps checkpoints associated with the active run without requiring every experiment
to specify an explicit checkpoint directory.

For example:

```text
workspace/
└── hydra_logs/
    └── runs/
        └── 2026-08-09/
            └── 10-41-35/
                ├── .hydra/
                ├── train.log
                └── checkpoints/
                    ├── last.ckpt
                    └── epoch=42-step=30272.ckpt
```

The exact directory structure depends on the configured Hydra and infrastructure paths.

## Checkpoint Configuration

Checkpoint behavior is configured under the top-level `checkpoint` section.

Supported fields include:

* `enabled`
* `monitor`
* `mode`
* `save_last`
* `save_top_k`
* `every_n_epochs`
* `every_n_train_steps`
* `dirpath`
* `filename`

A typical configuration might be:

```yaml
enabled: true
monitor: val_loss
mode: min
save_last: true
save_top_k: 1
every_n_epochs: 1
```

The checkpoint configuration is passed into the callback factory, which constructs the
appropriate checkpoint callback.

## Common Checkpoint Patterns

### Save the Best Validation Checkpoint

For a loss metric:

```yaml
monitor: val_loss
mode: min
save_top_k: 1
save_last: true
```

For an accuracy metric:

```yaml
monitor: val_accuracy
mode: max
save_top_k: 1
save_last: true
```

### Save Several Best Checkpoints

Increase `save_top_k`:

```yaml
save_top_k: 3
```

This preserves the three best checkpoints according to the monitored metric.

### Save on an Epoch Cadence

Use:

```yaml
every_n_epochs: 5
```

This is useful when checkpoints are expensive or training runs for a large number of epochs.

### Save on a Step Cadence

Use:

```yaml
every_n_train_steps: 1000
```

This is useful for long-running training jobs where epoch boundaries are too infrequent.

### Disable Checkpointing

Set:

```yaml
enabled: false
```

This can be useful for short development or CI runs where checkpoint artifacts are not
required.

## Reproducibility

AIDE treats the resolved experiment configuration as part of the training artifact.

A training run has three important pieces of state:

1. **Configuration** — the resolved `ExperimentConfig`
2. **Data reference** — the dataset manifest used by the datamodule
3. **Model state** — Lightning checkpoints

The resolved configuration is logged to MLflow as:

```text
experiment_config.json
```

and checkpoints are persisted as MLflow artifacts.

This means a run is not represented only by a model checkpoint.

The configuration and dataset reference used to produce that checkpoint are also retained
with the experiment.

Conceptually:

```text
             Training Run
                  │
       ┌──────────┼──────────┐
       │          │          │
       ▼          ▼          ▼
    Config      Data       Model
       │       Reference    State
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
              MLflow Run
```

This is important for reproducibility because a checkpoint without the configuration that
produced it may not be sufficient to reproduce or meaningfully evaluate the experiment.

## MLflow Experiment Naming

The trainer wrapper currently constructs the MLflow logger with a fixed experiment name in
code.

The MLflow tracking location is configurable, but the MLflow experiment name is not currently
exposed as an `InfrastructureConfig` field.

This is a current implementation limitation and a natural future extension if multiple
logical MLflow experiments need to be managed from the same AIDE installation.

## Common Workflow

A typical training run follows this sequence:

```text
1. Load environment
        │
        ▼
2. Compose Hydra configuration
        │
        ▼
3. Validate with ExperimentConfig
        │
        ▼
4. Discover local and configured plugins
        │
        ▼
5. Construct model and datamodule
        │
        ▼
6. Construct MLflow logger
        │
        ▼
7. Construct callbacks
        │
        ▼
8. Log resolved configuration
        │
        ▼
9. Run Lightning Trainer
        │
        ▼
10. Write checkpoints
        │
        ▼
11. Persist artifacts to MLflow
```

This sequence is the primary orchestration responsibility of AIDE's training runtime.

## Why AIDE Uses Lightning Instead of Reimplementing Training

AIDE intentionally builds on Lightning rather than implementing its own training loop.

Lightning already provides mature implementations for:

* training and validation loops
* distributed execution
* accelerator selection
* precision management
* checkpointing
* callbacks
* logging
* lifecycle hooks
* optimizer handling

AIDE's role is to standardize how these capabilities are configured and connected to the
rest of the experiment infrastructure.

This allows users to retain Lightning's familiar programming model while gaining a
consistent project-level MLOps structure.

## Practical Guidance

Use the configuration groups according to their responsibilities:

* Put model architecture and model hyperparameters in `configs/model/`.
* Put data loading and augmentation behavior in `configs/datamodule/`.
* Put hardware and runtime behavior in `configs/trainer/`.
* Put checkpoint policy in `configs/checkpoint/`.
* Put tracking and storage concerns in `configs/infrastructure/`.
* Put stable combinations of these settings in `configs/experiment/`.

Prefer named trainer and experiment configurations for workflows that will be repeated.

For example:

```text
configs/
├── experiment/
│   ├── default.yaml
│   ├── gpu.yaml
│   └── smoke_test.yaml
│
├── trainer/
│   ├── default.yaml
│   ├── gpu.yaml
│   └── smoke_test.yaml
│
└── checkpoint/
    ├── default.yaml
    └── disabled.yaml
```

This makes common workflows explicit and version-controlled rather than hidden in long
command-line override strings.

## Design Principles

The trainer and checkpoint system follows several principles:

### Keep Training Semantics in Lightning

AIDE should configure and orchestrate Lightning rather than create a competing training API.

### Keep Infrastructure Out of Model Code

Models should implement task behavior.

They should not need to know:

* where MLflow is running
* where checkpoints are stored
* how artifacts are uploaded
* how Hydra run directories are organized

Those concerns belong to the training infrastructure.

### Treat Configuration as an Artifact

The resolved configuration is part of the experiment's state and should be retained with
the resulting model artifacts.

### Separate Working Storage from Experiment Storage

Local files are useful during execution.

MLflow provides a mechanism for associating persistent artifacts with an experiment run.

### Prefer Composition Over Duplication

Trainer, checkpoint, infrastructure, model, and datamodule configurations should remain
independently reusable and be composed into named experiments.

## Current Scope

The current implementation focuses on local training and MLflow-backed artifact persistence.

The architecture is intended to support more sophisticated infrastructure in the future,
including remote artifact stores and different deployment environments, without requiring
changes to the model programming interface.

The current implementation deliberately keeps the trainer wrapper small so that additional
infrastructure can be introduced without coupling experiment code directly to it.

## Related Pages

* `wikis/Configuration.md`
* `wikis/TrainableModel.md`
* `wikis/PluginsAndRegistries.md`
