# Experiments

This directory contains training scripts, configuration files, and experiment runners.

## Structure

```
experiments/
├── configs/           # YAML experiment configs
├── ssl/               # Self-supervised pretraining experiments
├── finetune/          # Supervised fine-tuning experiments
├── xai/               # Explainability evaluation experiments
└── fairness/          # Fairness auditing experiments
```

## Running Experiments

```bash
python experiments/ssl/pretrain.py --config experiments/configs/ssl_simclr.yaml
```
