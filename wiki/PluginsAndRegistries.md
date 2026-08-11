# Plugins and Registries

AIDE uses a plugin architecture to allow projects to add models, components, and transforms without modifying the AIDE framework itself.

The architecture separates four responsibilities:

```text
Plugin Discovery
       │
       ▼
Python Import
       │
       ▼
Registry Registration
       │
       ▼
Factory Construction
```

Configuration selects a registry key. The plugin system makes the corresponding implementation available. The factory then constructs the selected class.

This allows the training runtime to remain independent of the specific models and components used by an experiment.

---

## Why Registries Exist

AIDE does not construct project-specific implementations using hardcoded conditionals such as:

```python
if config.class_name == "cnn":
    return CNN(...)
elif config.class_name == "resnet":
    return ResNet(...)
```

Instead, implementations register themselves under stable keys:

```text
Configuration
     │
     │ class_name: my_model
     ▼
ModelRegistry
     │
     │ lookup("my_model")
     ▼
MyModel
```

This keeps the framework's construction logic generic while allowing individual projects to define their own implementations.

The three primary registries are:

* `ModelRegistry`
* `ComponentRegistry`
* `TransformRegistry`

---

## What `class_name` Means

The configuration field is named `class_name`, but it does **not** contain a Python class name or import path.

It contains a registry key.

For example:

```yaml
class_name: my_model
params:
  lr: 3e-4
```

The following is therefore valid:

```text
my_model
scaffold_cnn
my_preprocessor
scaffold_random_crop
```

A fully qualified Python path such as:

```text
my_project.models.MyModel
```

is not used.

At runtime, AIDE:

1. identifies the appropriate registry
2. looks up `class_name`
3. retrieves the registered class
4. passes `params` to the class constructor

This means configuration describes **what implementation should be used**, while the plugin provides **how that implementation works**.

---

## Registration

Models, components, and transforms use the same registration pattern.

### Models

```python
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(...): ...
```

### Components

```python
from aide.registry.registries import ComponentRegistry


@ComponentRegistry.register("my_component")
class MyComponent(...): ...
```

### Transforms

```python
from aide.registry.registries import TransformRegistry


@TransformRegistry.register("my_transform")
class MyTransform(...): ...
```

The decorator registers the class when the module is imported.

This is an important property of the system:

```text
import module
     │
     ▼
decorator executes
     │
     ▼
class enters registry
```

The registry therefore does not need to know where a class is physically located.

---

## Registry Type Safety

Registries are not completely untyped class containers.

Each registry has an expected base type.

Conceptually:

```text
ModelRegistry
    └── TrainableModel

ComponentRegistry
    └── Component

TransformRegistry
    └── TransformComponent
```

Therefore:

* models must satisfy the `TrainableModel` contract
* components must satisfy the `Component` contract
* transforms must satisfy the `TransformComponent` contract

This catches an important class of configuration mistakes before the object reaches the training runtime.

It also establishes a contract between the framework and user-provided implementations.

---

## Where User Code Lives

For scaffolded projects, project-specific implementations live under `plugins/`.

A typical project looks like:

```text
example/
├── configs/
│   ├── datamodule/
│   ├── experiment/
│   ├── infrastructure/
│   ├── model/
│   └── trainer/
└── plugins/
    ├── __init__.py
    ├── components/
    │   ├── __init__.py
    │   └── transforms.py
    └── models/
        ├── __init__.py
        └── cnn.py
```

The framework itself remains under the installed `aide` package.

This gives a useful separation:

```text
AIDE framework
    │
    ├── registries
    ├── factories
    ├── training runtime
    └── configuration contracts

Project
    │
    ├── configs
    └── plugins
         ├── models
         ├── components
         └── transforms
```

A project therefore extends AIDE rather than modifying AIDE.

---

## Discovery and Registration

Registration only happens when Python imports the module containing the registration decorator.

For example:

```python
@ModelRegistry.register("my_model")
class MyModel(...): ...
```

does nothing if the module containing `MyModel` is never imported.

Plugin discovery solves that problem.

The startup sequence is approximately:

```text
Start training
      │
      ▼
Load framework plugins
      │
      ▼
Load configured external packages
      │
      ▼
Discover local project plugins
      │
      ▼
Import plugin modules
      │
      ▼
Registration decorators execute
      │
      ▼
Registries populated
      │
      ▼
Construct configured objects
```

This distinction is important:

> **Discovery finds Python modules. Registration makes their classes available to the runtime.**

Discovery itself does not instantiate models or components.

---

## Framework-Owned Plugins

AIDE can load framework-owned implementations as part of its startup process.

These provide functionality that belongs to the framework itself and can be made available through the same registry mechanism used by project plugins.

The benefit is that the factory does not need separate construction logic for built-in versus user-defined implementations.

Both ultimately become registry entries.

---

## Local Project Plugins

The normal scaffolded workflow is to place project-specific code under:

```text
plugins/
```

AIDE discovers the local plugin directory associated with the project and imports its Python modules.

For example:

```text
plugins/models/cnn.py
```

may contain:

```python
@ModelRegistry.register("scaffold_cnn")
class CNNClassifier(...): ...
```

After discovery:

```text
scaffold_cnn
      │
      ▼
ModelRegistry
      │
      ▼
CNNClassifier
```

The experiment configuration can then select it:

```yaml
class_name: scaffold_cnn
```

No framework source code needs to change.

---

## Package Plugins

AIDE also supports plugins distributed as normal Python packages.

The infrastructure configuration can specify packages that should be imported during startup.

This provides a second extension model:

```text
Project-local implementation
        │
        ▼
plugins/

Shared implementation
        │
        ▼
Python package
```

Local plugins are appropriate when an implementation belongs to one experiment or project.

Package plugins are appropriate when an implementation should be shared across multiple projects or teams.

This distinction allows AIDE to evolve from a project-level framework into a reusable internal platform without requiring all custom code to live inside the AIDE repository.

---

## Factories

Once discovery has populated the registries, the factories do not need to know where implementations came from.

The model factory effectively performs:

```python
ModelRegistry.get(config.class_name)
```

The component factory performs:

```python
ComponentRegistry.get(config.class_name)
```

The transform factory performs:

```python
TransformRegistry.get(config.class_name)
```

The resulting class is then instantiated using the configured parameters.

Conceptually:

```text
                 Configuration
                      │
                      │
                class_name
                      │
                      ▼
                 Registry
                      │
                      │ get()
                      ▼
                 Python class
                      │
                    params
                      │
                      ▼
                 Instance
```

This is the main architectural payoff of the registry pattern.

The factories are stable even as the number of available implementations grows.

---

## Configuration → Registry → Runtime

Putting the pieces together, a configured model follows this path:

```text
configs/model/cnn.yaml
        │
        │ class_name: scaffold_cnn
        ▼
   Hydra composition
        │
        ▼
   ExperimentConfig
        │
        ▼
   Model factory
        │
        ▼
   ModelRegistry.get("scaffold_cnn")
        │
        ▼
   Registered CNN class
        │
        ▼
   Instantiate with params
        │
        ▼
   TrainableModel
        │
        ▼
   Lightning Trainer
```

This separation is deliberate.

Configuration controls **selection**.

Registries provide **lookup**.

Factories handle **construction**.

The training runtime handles **execution**.

---

## Adding a New Model

Adding a model requires only two conceptual changes.

### 1. Implement and register the class

```python
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(TrainableModel): ...
```

### 2. Select the registry key in configuration

```yaml
class_name: my_model
params:
  lr: 3e-4
```

The framework does not need a new `if` statement, factory branch, or import.

That is the primary extensibility mechanism of AIDE.

---

## Failure Modes

Because registration occurs during import, an unknown registry key usually means one of a few things went wrong.

### Unknown registry key

For example:

```text
KeyError: Unknown model: my_model
```

First check that:

1. the class is registered under the expected key
2. the module containing the decorator exists
3. the module was discovered
4. the module was successfully imported
5. the correct registry was used

### Registration collision

Registry keys should be treated as stable identifiers.

Avoid registering unrelated classes under the same key.

If a project intentionally replaces an existing implementation, that behavior should be explicit rather than accidental.

### Import failure

A plugin that raises an exception while being imported cannot register its classes.

Therefore plugin discovery errors should generally be treated as startup failures rather than silently ignored.

---

## Practical Guidance

Use registry keys as stable configuration-level identifiers.

Prefer:

```yaml
class_name: resnet_classifier
```

over implementation-specific identifiers such as:

```yaml
class_name: my_current_resnet_v2
```

Keep constructor parameters in configuration:

```yaml
class_name: resnet_classifier
params:
  lr: 3e-4
  num_classes: 10
```

rather than hardcoding experiment-specific values inside the plugin.

Use:

* local `plugins/` for project-specific implementations
* package plugins for implementations shared across projects
* registries for implementation lookup
* factories for object construction
* configuration for implementation selection and parameters

When a registry key is unknown, first verify that the module containing the registration decorator was actually imported.

---

## Architectural Summary

The plugin system is designed around a simple separation of concerns:

```text
┌──────────────────────┐
│      Configuration   │
│  "use my_model"      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Registry         │
│  "my_model" → class  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│       Factory        │
│   construct object   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     AIDE Runtime     │
│      train/evaluate  │
└──────────────────────┘
```

The resulting architecture allows the AIDE framework to remain stable while projects independently add models, data transforms, and other components.

That is the core purpose of the plugin and registry system: **extend the platform without modifying the platform.**

## Related Pages

* [Configuration](Configuration.md)
* [TrainableModel](TrainableModel.md)
* [Trainer and Checkpoints](TrainerAndCheckpoints.md)
* [Home](Home.md)
