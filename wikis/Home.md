# AIDE Wiki

These pages expand on the concepts introduced in the main README.

- `Configuration.md`: Hydra composition, config groups, experiments, and override patterns
- `TrainableModel.md`: how AIDE's trainable model base class works and how to extend it
- `TrainerAndCheckpoints.md`: Lightning trainer integration, MLflow logging, and checkpoint flow
- `PluginsAndRegistries.md`: plugin discovery, registry keys, and how config resolves to code

Suggested reading order:

1. README for installation and the scaffold workflow.
2. `Configuration.md` for Hydra composition.
3. `TrainableModel.md` for model extension.
4. `TrainerAndCheckpoints.md` for runtime and artifact behavior.
5. `PluginsAndRegistries.md` for code registration and discovery details.