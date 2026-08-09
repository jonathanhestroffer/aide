# Plugins and Registries

This page explains how AIDE discovers user code and how config values resolve to actual Python
classes.

## Why Registries Exist

AIDE does not import models or transforms by hardcoded `if` statements. Instead, it uses
registries keyed by short names.

The three main registries are:

- `ModelRegistry`
- `ComponentRegistry`
- `TransformRegistry`

This lets configuration choose behavior by key while keeping construction logic generic.

## What `class_name` Means in AIDE

The config field is called `class_name`, but in practice it stores a registry key.

Examples:

- `scaffold_cnn`
- `scaffold_random_crop`
- `my_model`
- `my_preprocessor`

It is not a fully qualified Python path such as `my_project.models.MyModel`.

At runtime, AIDE uses that key to look up the class in the relevant registry and then passes the
`params` mapping into the class constructor.

## Registration Pattern

Models:

```python
from aide.registry.registries import ModelRegistry


@ModelRegistry.register("my_model")
class MyModel(...): ...
```

Components:

```python
from aide.registry.registries import ComponentRegistry


@ComponentRegistry.register("my_component")
class MyComponent(...): ...
```

Transforms:

```python
from aide.registry.registries import TransformRegistry


@TransformRegistry.register("my_transform")
class MyTransform(...): ...
```

Each registry validates the class type:

- models must inherit from `TrainableModel`
- components must inherit from `Component`
- transforms must inherit from `TransformComponent`

## Where User Code Lives

For scaffolded projects, custom code lives under `plugins/`.

Typical structure:

```text
plugins/
  __init__.py
  models/
  components/
```

The scaffold already provides this layout.

## How Discovery Works

When AIDE loads plugins, it imports:

1. framework-owned built-in packages such as `aide.models` and `aide.components`
2. user-specified packages from config, if any
3. Python files discovered under the local `plugins/` directory

Decorator-based registration happens at import time. That means importing a module is enough to
populate the relevant registry.

For a scaffolded project, the local plugin directory is resolved from `.env` and discovered
automatically by the project launcher path.

## Two Plugin Paths

There are two ways user code can be brought in.

### Local Scaffold Plugins

When you use the generated project structure, AIDE loads the local `plugins/` directory and
imports its Python files recursively.

This is the normal path for scaffolded projects.

### Package Plugins Listed in Config

The infrastructure config also has a `plugins` field. Those entries are package names that AIDE
will import explicitly.

Use this when custom code lives in a separately installed Python package rather than the local
project directory.

## How Factories Use Registries

Once plugin discovery is done, AIDE's factories stay simple.

- model factory: `ModelRegistry.get(config.class_name)`
- component factory: `ComponentRegistry.get(config.class_name)`
- transform factory: `TransformRegistry.get(config.class_name)`

This is why most customization work has two steps only:

1. register a class under a stable key
2. refer to that key in config

## Practical Guidance

- Keep registry keys short and stable.
- Put constructor parameters in config, not in hardcoded launch scripts.
- Use local `plugins/` for project-specific code.
- Use config-listed packages when code should be shared across projects as a Python package.
- If AIDE says a key is unknown, first verify that the module containing the decorator was
  actually imported.

## Related Pages

- `wikis/Configuration.md`
- `wikis/TrainableModel.md`
- `wikis/TrainerAndCheckpoints.md`