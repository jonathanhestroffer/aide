# TrainableModel

`TrainableModel` is AIDE's contract for trainable models.

It extends Lightning's `LightningModule` rather than replacing it. AIDE therefore uses Lightning for the actual training lifecycle while adding a small amount of framework-specific structure around model selection, configurable components, and reproducible configuration.

If you know how to implement a Lightning `LightningModule`, the AIDE model contract should feel familiar.

---

## What `TrainableModel` Provides

The relationship is:

```text id="h2p5ag"
                 LightningModule
                       │
                       ▼
                TrainableModel
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   AIDE model contract       Configurable pipeline
          │                         │
          │                  preprocessor
          │                         │
          │                       model
          │                         │
          │                  postprocessor
          │
          ▼
    Project model
```

AIDE adds two optional pipeline components:

* `self.preprocessor`
* `self.postprocessor`

Both default to identity components.

The model factory can replace these with configured components before training begins.

This allows a model to remain focused on task-specific behavior while preprocessing and postprocessing can be selected independently through configuration.

---

## The Model Contract

`TrainableModel` is an abstract `LightningModule`.

A concrete model must implement the methods required by the AIDE contract:

* `forward`
* `training_step`
* `configure_optimizers`

Standard Lightning lifecycle hooks remain available as well.

For example:

* `validation_step`
* `test_step`
* `predict_step`
* `on_train_start`
* `on_validation_epoch_end`

AIDE does not change the semantics of these Lightning hooks.

The underlying contract remains Lightning's.

---

## Minimal Model

A typical AIDE model looks like:

```python id="y3wqzi"
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(TrainableModel):
    def __init__(
        self,
        hidden_dim: int = 128,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        # Build the actual network here.
        ...

    def forward(self, x):
        x = self.preprocessor(x)

        output = ...

        return self.postprocessor(output)

    def training_step(self, batch, batch_idx): ...

    def configure_optimizers(self): ...
```

The important point is that the model remains a normal Lightning model.

The registry decorator makes it discoverable by AIDE, while `TrainableModel` provides the common contract.

---

## The Configurable Model Pipeline

AIDE allows the model pipeline to be decomposed into three stages:

```text id="k5t4xb"
Input
  │
  ▼
┌─────────────────┐
│  Preprocessor   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Model      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Postprocessor  │
└────────┬────────┘
         │
         ▼
      Output
```

The preprocessor and postprocessor default to identity operations, so a model does not need to configure them.

When configured, AIDE constructs them through `ComponentRegistry`.

This allows behavior such as preprocessing or output decoding to be changed without creating a new model class for every combination.

---

## How the Model Is Constructed

Model construction follows the same configuration → registry → factory pattern used throughout AIDE.

A model configuration might contain:

```yaml id="5mgvqs"
class_name: scaffold_cnn
params:
  num_classes: 10
  lr: 1e-3
```

AIDE resolves:

```text id="y2t5s9"
trainable.model.class_name
          │
          ▼
   ModelRegistry
          │
          ▼
    scaffold_cnn
          │
          ▼
     CNN class
          │
          ▼
trainable.model.params
          │
          ▼
    model instance
```

The value of `class_name` is a registry key, not a Python import path.

This keeps model selection independent from the physical location of the implementation.

---

## Configured Components

The same model configuration can optionally specify components.

Conceptually:

```yaml id="4qjzfr"
trainable:
  model:
    class_name: scaffold_cnn
    params:
      num_classes: 10

  preprocessor:
    class_name: my_preprocessor
    params:
      ...

  postprocessor:
    class_name: my_postprocessor
    params:
      ...
```

AIDE resolves the model through `ModelRegistry` and the components through `ComponentRegistry`.

The resulting runtime object therefore becomes:

```text id="0r8q1j"
             TrainableModel
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 Preprocessor     Model    Postprocessor
```

This separation is useful when the same preprocessing or postprocessing behavior should be reused by multiple models.

---

## Model vs Component vs Transform

AIDE provides three different extension points.

### Model

Use a model when the logic represents the trainable task or network itself.

Examples:

* CNN classifier
* object detector
* segmentation model
* sequence model
* GAN
* diffusion model

### Component

Use a component when the logic is a reusable part of the model pipeline.

Examples:

* input preprocessing
* feature adaptation
* output decoding
* prediction postprocessing
* reusable model-side transformations

### Transform

Use a transform when the logic belongs to the dataset pipeline.

Examples:

* image augmentation
* normalization
* cropping
* dataset-specific preprocessing

The distinction can be summarized as:

```text id="q2c6jg"
Dataset
   │
   ▼
Transform
   │
   ▼
Datamodule
   │
   ▼
Preprocessor
   │
   ▼
TrainableModel
   │
   ▼
Postprocessor
   │
   ▼
Prediction
```

This keeps data preparation, model computation, and output handling as separate extension points.

---

## Reproducibility

`TrainableModel` should treat constructor arguments as part of the experiment definition.

For example:

```python id="y8j6so"
def __init__(
    self,
    hidden_dim: int = 128,
    lr: float = 1e-3,
):
    super().__init__()
    self.save_hyperparameters()
```

Using `save_hyperparameters()` allows Lightning to retain the model's constructor configuration as part of the model state and experiment metadata.

This is particularly useful when combined with AIDE's configuration and MLflow artifact tracking.

The resulting provenance can be thought of as:

```text id="g6t2ce"
Experiment Config
       │
       ▼
Model Parameters
       │
       ▼
Lightning Model
       │
       ├── Metrics
       ├── Checkpoints
       └── Experiment Artifacts
```

The goal is that a checkpoint is not just a set of learned weights; it exists within a recorded experiment configuration.

---

## Logging

Models should use Lightning's logging interface:

```python id="3u7x4b"
self.log("train_loss", loss)
self.log("val_loss", loss)
```

rather than directly coupling model code to MLflow.

AIDE's trainer and logger infrastructure is responsible for connecting Lightning's logging behavior to the configured experiment tracking backend.

This keeps the model implementation independent of the particular logging infrastructure.

For example, the model should not need to know whether the experiment is being tracked locally or through a remote MLflow server.

---

## Optimization

`configure_optimizers()` follows Lightning's standard optimizer contract.

For example:

```python id="y5nq3d"
def configure_optimizers(self):
    return torch.optim.Adam(
        self.parameters(),
        lr=self.lr,
    )
```

More advanced models can use Lightning's supported optimizer patterns, including multiple optimizers and manual optimization where appropriate.

AIDE does not introduce a separate optimization API.

The model remains responsible for defining its optimization behavior.

---

## Lifecycle and Hooks

AIDE does not redefine the Lightning lifecycle.

For example:

```text id="4k3w8q"
Trainer.fit()
    │
    ▼
Lightning lifecycle
    │
    ├── training_step()
    ├── validation_step()
    ├── optimizer handling
    ├── callbacks
    ├── logging
    └── checkpointing
```

AIDE's role is primarily to construct and configure the objects that participate in that lifecycle.

This is an intentional design decision.

Rather than building another training abstraction on top of Lightning, AIDE provides the project-level infrastructure around Lightning.

---

## Best Practices

### Keep the Model Focused

The model should contain task-specific behavior.

Avoid putting experiment wiring, artifact management, or MLflow-specific code into the model.

### Use Configuration for Experiment Parameters

Prefer:

```yaml id="rj4s5u"
params:
  lr: 3e-4
```

over hardcoding experiment-specific values.

### Use Components for Reusable Pipeline Logic

If preprocessing or postprocessing should be configurable or shared between models, make it a component rather than duplicating it inside each model.

### Use Lightning's APIs

Use:

* `self.log`
* `configure_optimizers`
* Lightning lifecycle hooks
* Lightning checkpointing
* Lightning optimization APIs

AIDE is designed to work with Lightning rather than hide it.

### Preserve Reproducibility

Use `save_hyperparameters()` when constructor parameters materially affect model behavior.

Keep experiment-specific values in configuration so they can be recorded alongside training artifacts.

---

## What AIDE Does Not Abstract

AIDE intentionally does not try to replace Lightning's model API.

The following remain Lightning responsibilities:

* optimizer semantics
* training lifecycle
* distributed training
* mixed precision
* gradient handling
* manual optimization
* checkpoint callbacks
* training hooks
* validation and test lifecycle

AIDE provides the surrounding platform:

```text id="4h0x9j"
                AIDE
     ┌────────────┼────────────┐
     ▼            ▼            ▼
 Configuration  Registry    Infrastructure
     │            │            │
     └────────────┼────────────┘
                  ▼
            LightningModule
                  │
                  ▼
             Lightning
```

This keeps the framework relatively thin while providing a consistent project-level architecture.

---

## Summary

`TrainableModel` is the contract between an AIDE project and the Lightning training runtime.

A project supplies the model implementation.

AIDE supplies:

* registry-based discovery
* configuration-driven construction
* optional model pipeline components
* common training infrastructure
* integration with experiment tracking and artifacts

The resulting separation is:

```text id="z1s2t8"
Project Code
     │
     │ implements
     ▼
TrainableModel
     │
     │ constructed by
     ▼
AIDE Registry + Factory
     │
     │ configured by
     ▼
Hydra + Pydantic
     │
     │ executed by
     ▼
Lightning
```

The primary design principle is simple:

> **AIDE owns the platform; the project owns the model.**

## Related Pages

* [Configuration](Configuration.md)
* [Plugins and Registries](PluginsAndRegistries.md)
* [Trainer and Checkpoints](TrainerAndCheckpoints.md)
* [Home](Home.md)
