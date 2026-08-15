# Reproducibility

Publication-facing experiments should be recoverable without relying on memory or an external dashboard.

For each run, create one row in `experiment_manifest.csv` and preserve the corresponding configuration and result artifacts.

Recommended run outputs:

```text
results/runs/<run_id>/
├── config.yaml
├── environment.txt
├── metrics.json
├── predictions.csv
├── training_history.csv
├── split_manifest.json
└── figures/
```

A run ID should be stable and human-readable, for example:

```text
E01_densenet121_asl_seed42
E02_resnet50_asl_seed42
E03_densenet121_bce_seed42
```

Do not overwrite a completed run directory. If a run is repeated after a code/config change, create a new run ID and record the new commit SHA.

## Minimum metadata

- experiment/run ID
- date
- Git commit SHA
- backbone
- loss
- initialization/pretraining
- image size
- batch size
- optimizer and learning rate
- epoch/early-stopping settings
- seed
- train/validation/test split identifier or hash
- checkpoint path or external artifact ID
- raw prediction path
- metric path
- notes/status
