# Trainer and Checkpoints

This page explains how AIDE wraps Lightning's trainer and how checkpointing is configured.

## Trainer Overview

AIDE uses a small wrapper class around `lightning.Trainer`.

Its responsibilities are:

- create the MLflow-backed logger
- build configured callbacks
- pass typed trainer settings into Lightning
- log the resolved experiment config as an artifact
- run `trainer.fit(...)`

The wrapper is intentionally small. The goal is to centralize boilerplate, not replace
Lightning's core behavior.

## Trainer Configuration

The scaffolded trainer config controls a subset of Lightning trainer options directly:

- `max_epochs`
- `accelerator`
- `devices`
- `precision`

These come from `configs/trainer/` and are forwarded into `lightning.Trainer`.

This means hardware selection remains a config concern rather than something hidden in model
code or handwritten launch scripts.

Typical example:

```yaml
max_epochs: 100
accelerator: auto
devices: auto
precision: "32"
```

Create additional trainer configs if you want named runtime profiles such as:

- CPU debug
- single-GPU training
- mixed precision
- shorter smoke tests

## MLflow Integration

AIDE creates an `MLFlowLoggerAdapter` and attaches it to Lightning.

That logger does two things:

1. It works as a normal Lightning logger for metrics.
2. It exposes direct MLflow client helpers such as `log_dict`, `log_artifact`, and
   `log_artifacts`.

Before training starts, AIDE logs the resolved `ExperimentConfig` as `experiment_config.json`.
That gives each run a complete config snapshot in artifact storage.

At the moment, the trainer wrapper constructs the MLflow logger with a fixed experiment name in
code. Tracking location is configurable; the MLflow experiment name is not yet surfaced as a
config field.

## Output and Tracking Configuration

These values live under `infrastructure` config:

- `tracking_uri`
- `artifact_location`
- `save_dir`

In the scaffold they default to local paths rooted under the project workspace, but they can be
changed with config or environment variables.

Use this split as a rule of thumb:

- `trainer` decides how training runs
- `infrastructure` decides where outputs and tracking data go

## Checkpoint Flow

Checkpoint behavior is configured under the top-level `checkpoint` section, which is composed
from the checkpoint config group.

Supported fields include:

- `enabled`
- `monitor`
- `mode`
- `save_last`
- `save_top_k`
- `every_n_epochs`
- `every_n_train_steps`
- `dirpath`
- `filename`

AIDE builds an `ArtifactModelCheckpoint`, which subclasses Lightning's `ModelCheckpoint`.

## What AIDE's Checkpoint Callback Adds

Lightning already knows how to save checkpoints locally. AIDE extends that behavior by uploading
the final checkpoint directory into MLflow artifact storage at the end of training.

That means checkpoint flow has two stages:

1. Lightning writes checkpoints locally.
2. AIDE uploads that checkpoint directory to MLflow artifacts.

If artifact upload fails, AIDE intentionally does not fail training cleanup.

## How Checkpoint Paths Are Chosen

If `checkpoint.dirpath` is not set:

- when Hydra is initialized, AIDE uses Hydra's runtime output directory and appends
  `checkpoints/`
- otherwise it falls back to `infrastructure.save_dir/checkpoints`

This keeps checkpoint paths aligned with the active run directory without requiring every
experiment to set an explicit checkpoint path.

## Common Customization Patterns

### Save the Best Validation Checkpoint

Use:

```yaml
monitor: val_loss
mode: min
save_top_k: 1
save_last: true
```

### Save Several Best Checkpoints

Increase `save_top_k`.

### Checkpoint on Epoch Cadence

Set `every_n_epochs`.

### Checkpoint on Step Cadence

Set `every_n_train_steps`.

### Disable Checkpointing

Set:

```yaml
enabled: false
```

## Practical Guidance

- Put runtime behavior in `configs/trainer/`.
- Put output, tracking, and checkpoint behavior in `configs/infrastructure/` and
  `configs/checkpoint/`.
- Prefer named trainer profiles over long CLI override strings for common workflows.
- Let MLflow hold the final run config and checkpoint artifacts for reproducibility.

## Related Pages

- `wikis/Configuration.md`
- `wikis/TrainableModel.md`
- `wikis/PluginsAndRegistries.md`