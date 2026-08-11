# AIDE

**AIDE** is the **AI Development & Engineering Platform**.

AIDE is a configuration-driven ML development platform for standardizing experimentation and training across teams. It provides a common project structure, configuration system, extension model, training runtime, experiment tracking, and artifact management so ML engineers can focus on models and data rather than rebuilding training infrastructure for every project.

AIDE is built around [PyTorch Lightning](https://lightning.ai/) and [MLflow](https://mlflow.org/), with [Hydra](https://hydra.cc/) for configuration composition and Pydantic for runtime configuration validation.

## What AIDE Handles

* **Project scaffolding** — create standardized ML experiment projects without rebuilding boilerplate
* **Configuration-driven experiments** — compose experiments from reusable Hydra configuration groups
* **Typed configuration contracts** — validate resolved experiment configuration with Pydantic
* **Plugin architecture** — register and discover project-specific models, components, and transforms
* **Unified training runtime** — run projects through a shared PyTorch Lightning trainer interface
* **Experiment tracking** — record metrics, configuration, checkpoints, and artifacts with MLflow
* **Artifact management** — provide consistent locations for datasets, checkpoints, and experiment outputs
* **Developer workflow** — provide a common CLI for initializing, inspecting, and running ML projects

## Design Goals

AIDE is designed around four principles.

### Standardize the infrastructure, not the model

ML engineers should be able to implement different model architectures while sharing the same training, logging, checkpointing, configuration, and artifact infrastructure.

### Configuration over code

Experiment behavior should generally be changed through configuration rather than by modifying framework code.

### Explicit extension points

Project-specific models, components, and transforms are implemented through defined contracts and registries rather than modifying AIDE's core runtime.

### Reproducible experiments

Resolved configuration, metrics, checkpoints, and artifacts should remain associated with an experiment so that the training environment and decisions behind a result can be reconstructed.

## Architecture

AIDE separates platform-owned infrastructure from project-owned ML code.

```text
                         ┌──────────────────────┐
                         │         AIDE         │
                         │ ML Development &     │
                         │ Engineering Platform │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
            Configuration       Registries          CLI
            Hydra/Pydantic      Plugin Discovery    init/list/train
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                            Experiment Runtime
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                 Model          DataModule        Trainer
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                           PyTorch Lightning
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                    MLflow                 Checkpoints
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                                 Artifacts
```

The framework owns experiment orchestration, configuration, logging, checkpointing, and artifact management. Individual projects own models, transforms, components, and other domain-specific ML code.

This allows multiple projects to share the same engineering infrastructure without forcing them to share the same model implementation.

## Quick Start

Install AIDE from the repository root:

```bash
uv sync --frozen
source .venv/bin/activate
```

Create a scaffolded experiment project:

```bash
aide init example
cd example
```

This creates a project with:

```text
example/
  .env
  configs/
  plugins/
```

By default, `aide init` uses the current working directory as the shared artifact location and points `AIDE_DATASET_MANIFEST` at:

```text
./cifar10/manifest.json
```

If multiple projects should share a different dataset location, pass `--artifact-dir`:

```bash
aide init example --artifact-dir /shared/aide/artifacts
```

The generated project can then run the default experiment with:

```bash
aide train --experiment default
```

AIDE handles the experiment lifecycle:

1. Load the project environment
2. Resolve the experiment configuration
3. Discover and load project plugins
4. Validate the resolved configuration
5. Construct the model and datamodule
6. Configure the shared Lightning training runtime
7. Track the experiment with MLflow
8. Persist checkpoints and experiment artifacts

The model and dataset implementation are project-specific; the surrounding engineering infrastructure is provided by AIDE.

## How Configuration Works

AIDE uses Hydra config composition rather than one large hand-written configuration file.

The scaffold provides configuration groups such as:

```text
configs/
├── config.yaml
├── checkpoint/
├── datamodule/
├── experiment/
├── infrastructure/
├── model/
└── trainer/
```

These configuration groups separate concerns while allowing an experiment to compose a complete runtime configuration.

### Hydra

Hydra is responsible for configuration composition and experiment selection.

For example, the default experiment is implemented as:

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

This composes the experiment from independent configuration groups.

### Pydantic

After Hydra resolves the configuration, AIDE converts it into a typed `ExperimentConfig`.

This gives AIDE a clear runtime configuration contract:

```text
YAML configuration
        │
        ▼
      Hydra
   composition
        │
        ▼
Resolved configuration
        │
        ▼
     Pydantic
    validation
        │
        ▼
 ExperimentConfig
        │
        ▼
     Runtime
```

Hydra therefore handles **composition**, while Pydantic establishes the **validated runtime contract** consumed by the framework.

From the resulting `ExperimentConfig`:

* `trainable.model` is built from `ModelRegistry`
* `datamodule` is built from the artifact manifest path in configuration
* `trainer` fields are passed to the shared Lightning trainer
* `checkpoint` controls checkpoint cadence, naming, and monitored metrics
* `infrastructure` controls MLflow tracking, artifact storage, and output locations

The practical workflow is simple: change configuration to select behavior, and only write Python when a new model, transform, or component is required.

For full Hydra overrides beyond selecting `--experiment`, use the lower-level training entrypoint instead of the generated `train.py` wrapper. The wrapper is intentionally limited to selecting the experiment.

## Add a Model or Custom Component

Custom project code lives in the generated project's `plugins/` package.

To add a model:

1. Create a class that inherits from `TrainableModel`.
2. Register it with `ModelRegistry`.
3. Point the model configuration at the registered key.

```python
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(TrainableModel): ...
```

Then select it in Hydra configuration by setting the model's `class_name` to `my_model`.

The same extension pattern is used for other project-specific components:

* `ComponentRegistry` for preprocessors and postprocessors used by a model
* `TransformRegistry` for dataset transforms applied by the datamodule

At startup, AIDE imports the local plugin package before construction so registration side effects populate the registries. Factories then instantiate the configured classes from their `class_name` and `params` fields.

This creates a clear boundary between the shared platform and project-specific implementation:

```text
AIDE Platform
────────────────────────────
Configuration
Training runtime
Logging
Checkpointing
Artifacts
Registry infrastructure
        │
        │ stable extension contracts
        ▼
Project
────────────────────────────
Models
Transforms
Components
Experiment configuration
```

## Experiment and Artifact Lifecycle

AIDE associates experiment configuration and training outputs so that the result of an experiment can be traced back to the configuration that produced it.

```text
Experiment configuration
          │
          ▼
       Training
          │
     ┌────┼────┐
     ▼    ▼    ▼
 Metrics Config Checkpoints
     │    │    │
     └────┼────┘
          ▼
        MLflow
          │
          ▼
       Artifacts
```

By default, a scaffolded project writes runtime outputs to `workspace/`:

* `workspace/hydra_logs/` for Hydra run metadata
* `workspace/lightning_logs/` for Lightning outputs
* `workspace/metadata/aide.db` for local MLflow tracking
* `workspace/artifacts/` for MLflow artifacts such as configuration snapshots and checkpoints

Override these locations with environment variables such as:

```text
AIDE_TRACKING_URI
AIDE_ARTIFACT_LOCATION
AIDE_SAVE_DIR
```

The example project uses CIFAR-10 as a reference workload. The dataset and model are intentionally simple; the purpose of the example is to demonstrate AIDE's project structure, configuration system, plugin architecture, training runtime, and artifact lifecycle rather than model performance.

## Commands

Initialize a project:

```bash
aide init <experiment-name> [--artifact-dir <dir>]
```

List available project extensions:

```bash
aide list <project-path> [--kind models|components|transforms]
```

Run an experiment:

```bash
aide train --experiment <name>
```

For a scaffolded project, run the experiment from the project root:

```bash
aide train --experiment default
```

This command loads the project's `.env`, resolves `AIDE_CONFIGS_PATH`, discovers local plugins, and uses the project's `configs/` directory automatically.

For raw Hydra overrides or direct access to the lower-level training entrypoint, source the project environment and provide an explicit config path:

```bash
set -a
source .env
set +a

python -m aide.scripts.train \
    --config-path "$PWD/configs" \
    --config-name experiment/default \
    trainer.max_epochs=5
```

## Project Structure

A scaffolded AIDE project separates configuration and project-specific code:

```text
example/
├── .env
├── configs/
│   ├── checkpoint/
│   ├── datamodule/
│   ├── experiment/
│   ├── infrastructure/
│   ├── model/
│   └── trainer/
└── plugins/
    ├── components/
    └── models/
```

The generated project owns the experiment-specific pieces while AIDE provides the common runtime.

## MLOps Workflow

AIDE currently focuses on the experiment and training portion of the ML lifecycle:

```text
Project Initialization
        │
        ▼
Configuration
        │
        ▼
Plugin Discovery
        │
        ▼
Experiment Validation
        │
        ▼
Training
        │
        ├──────────────┐
        ▼              ▼
     Metrics       Checkpoints
        │              │
        └──────┬───────┘
               ▼
             MLflow
               │
               ▼
            Artifacts
```

The current implementation is intentionally focused on the development and experiment-management layer. Evaluation, model promotion, deployment, and production inference are natural extensions of this lifecycle rather than prerequisites for the core framework.

## Scope

The current reference implementation is centered on a local artifact-backed datamodule and a CIFAR-10 example.

The example is intended to demonstrate the framework architecture rather than provide a production-scale ML workload. The important parts of the example are the standardized project structure, configuration composition, plugin system, training runtime, experiment tracking, and artifact management.

## Further Reading

For deeper implementation details, see:

* `wikis/Home.md`
* `wikis/Configuration.md`
* `wikis/TrainableModel.md`
* `wikis/TrainerAndCheckpoints.md`
* `wikis/PluginsAndRegistries.md`
