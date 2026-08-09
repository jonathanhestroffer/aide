# TrainableModel

This page explains how AIDE's `TrainableModel` works and how to extend it correctly.

## What It Is

`TrainableModel` is AIDE's base class for trainable models. It inherits from Lightning's
`LightningModule`, so the contract is still the Lightning contract.

That means AIDE does not replace Lightning training semantics. It wraps them with:

- registry-based model selection
- optional preprocessor and postprocessor components
- typed config-driven construction

If you know how to write a `LightningModule`, you already know most of what matters here.

## What AIDE Adds

`TrainableModel` initializes two component hooks:

- `self.preprocessor`
- `self.postprocessor`

Both default to identity components. The model factory may replace them from config before
training starts.

This lets the model pipeline be configured as:

- preprocessor
- model forward path
- postprocessor

without hardcoding those pieces in the model class itself.

## Required Methods

Because `TrainableModel` is abstract, subclasses must implement:

- `forward`
- `training_step`
- `configure_optimizers`

You can also override standard Lightning hooks such as:

- `validation_step`
- `test_step`
- `predict_step`
- `on_train_start`
- `on_validation_epoch_end`

Follow Lightning's documentation for the semantics of these methods. AIDE does not change how
Lightning calls them.

## Minimal Shape

Typical model structure:

```python
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(TrainableModel):
    def __init__(self, hidden_dim: int = 128, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

    def forward(self, x):
        x = self.preprocessor(x)
        logits = ...
        return self.postprocessor(logits)

    def training_step(self, batch, batch_idx): ...

    def configure_optimizers(self): ...
```

The important detail is that your model should treat `preprocessor` and `postprocessor` as part
of the model pipeline if you want those extension points to remain useful.

## How the Model Is Constructed

At runtime, AIDE takes `trainable.model.class_name` from config and resolves it through
`ModelRegistry`. It instantiates the class with `trainable.model.params`.

In practice, `class_name` is a registry key, not an import path. A scaffolded model uses a key
such as `scaffold_cnn`.

If `trainable.preprocessor` or `trainable.postprocessor` are configured, AIDE builds those from
`ComponentRegistry` and attaches them to the model instance.

This is why model code should stay focused on actual model behavior while configuration controls
which optional components wrap it.

## When to Use a Model vs a Component

Use a model when the logic belongs in the trainable network or task itself.

Use a component when the logic is better treated as a configurable reusable unit, for example:

- input preprocessing
- output decoding
- reusable feature adaptation
- postprocessing before metric calculation or prediction export

Use a transform when the logic belongs at the dataset level rather than inside the model.

## Best Practices

- Call `super().__init__()` first.
- Use `save_hyperparameters()` when constructor arguments matter for reproducibility.
- Keep the model class focused on task logic, not experiment wiring.
- Log with Lightning's `self.log` so trainer and logger behavior stay consistent.
- Let config choose the registered class and its parameters.
- Reuse components for preprocessors and postprocessors when you want configurable behavior.

## Lightning Guidance Still Applies

For lifecycle details, optimizer return formats, manual optimization, distributed behavior, and
hook semantics, follow the Lightning `LightningModule` documentation. AIDE builds on top of that
API rather than inventing a separate one.

## Related Pages

- `wikis/Configuration.md`
- `wikis/TrainerAndCheckpoints.md`
- `wikis/PluginsAndRegistries.md`