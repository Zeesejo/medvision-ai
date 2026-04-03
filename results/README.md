# Results

Experiment outputs — checkpoints, logs, metrics.

> ⚠️ Model checkpoints and raw logs are excluded from git. Use Weights & Biases (wandb) for experiment tracking.

## Structure

```
results/
├── checkpoints/     # .pt model files (gitignored)
├── logs/            # Training logs (gitignored)
├── metrics/         # CSV/JSON metric summaries (tracked)
└── figures/         # Saved evaluation plots (tracked)
```
