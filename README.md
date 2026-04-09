 > [!WARNING]
> ## ⚠️ Work in Progress — Unpublished Research
>
> **This repository contains active, ongoing research for an unpublished academic paper.**
>
> - The code, methodology, models, and results are **incomplete and subject to major changes**
> - The associated paper has **not yet been peer-reviewed or submitted**
> - **Do not fork, clone, cite, or build upon this work** without explicit written permission from the author
> - Unauthorized use of this research may constitute academic misconduct
>
> 📬 For collaboration or inquiries, contact: **[Zeeshan Modi](https://github.com/Zeesejo)**

---

# 🩺 MedVision AI

> **Medical Computer Vision Research Project**  
> Self-supervised learning, Explainable AI, and Fairness in Medical Imaging  
> University of Bremen — M.Sc. AI/IS

[![CI](https://github.com/Zeesejo/medvision-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeesejo/medvision-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)

---

## 🎯 Project Overview

MedVision AI is a research codebase for multi-label chest pathology classification on the **NIH ChestX-ray14** dataset (112,120 frontal-view X-rays, 14 disease labels). The project targets publication at **MICCAI, MIDL, or ISBI** and focuses on three pillars:

- **Label efficiency** — self-supervised pretraining (SimCLR / BYOL) to reduce annotation cost
- **Explainability** — GradCAM + SHAP prototype explanations for clinical trust
- **Fairness** — demographic subgroup auditing to surface and mitigate bias

---

## 📁 Repository Structure

```
medvision-ai/
├── src/
│   ├── dataset.py          # ChestXrayDataset, class weights, transforms
│   ├── train.py            # Main training loop (AMP, mixed precision)
│   ├── evaluate.py         # Inference, AUC computation, ROC curves
│   ├── losses.py           # WeightedBCE, FocalLoss, AsymmetricLoss (ASL)
│   └── models/             # Backbone wrappers (ResNet, DenseNet, ViT)
├── tests/                  # Pytest unit tests
├── experiments/            # Config YAMLs and ablation results
├── docs/                   # Research notes and experiment logs
├── .github/
│   ├── workflows/ci.yml    # Lint + test on every PR
│   └── ISSUE_TEMPLATE/     # Bug, feature, experiment templates
├── CONTRIBUTING.md
└── README.md
```

---

## ⚙️ Installation

### Option 1 — Conda (recommended)

```bash
git clone https://github.com/Zeesejo/medvision-ai.git
cd medvision-ai
conda env create -f environment.yml
conda activate medvision
```

### Option 2 — pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🗂️ Dataset Setup

1. Download **NIH ChestX-ray14** from the [official source](https://nihcc.app.box.com/v/ChestXray-NIHCC) or via the NIH CLI tool.
2. Place files in the following structure:

```
data/
├── images/
│   ├── 00000001_000.png
│   └── ...
├── Data_Entry_2017.csv
├── train_val_list.txt
└── test_list.txt
```

3. Update `data_dir` in your config YAML (see `experiments/default.yaml`).

---

## 🏋️ Training

```bash
# Train with default config (ResNet-50, ASL loss, 30 epochs)
python src/train.py --config experiments/default.yaml

# Train with DenseNet-121
python src/train.py --config experiments/default.yaml --backbone densenet121

# Train with ViT-Base
python src/train.py --config experiments/default.yaml --backbone vit_base_patch16_224

# Disable AMP (e.g., CPU debug)
python src/train.py --config experiments/default.yaml --no-amp
```

Checkpoints are saved to `results/checkpoints/` and metrics are logged to **Weights & Biases** if `wandb.enabled: true` in config.

---

## 📊 Evaluation

```bash
# Run inference and generate per-class AUC + ROC curves
python src/evaluate.py --checkpoint results/checkpoints/best.pth --config experiments/default.yaml
```

Outputs written to `results/`:
- `roc_curves.png` — per-class ROC grid
- `auc_summary.json` — per-class and mean AUC
- `predictions.npy` — raw probabilities for further analysis

---

## 📈 Results

> **Baseline results on NIH ChestX-ray14 test split (ResNet-50, ASL, 30 epochs)**

| Pathology | AUC |
|---|---|
| Atelectasis | 0.814 |
| Cardiomegaly | 0.891 |
| Effusion | 0.883 |
| Infiltration | 0.709 |
| Mass | 0.839 |
| Nodule | 0.763 |
| Pneumonia | 0.762 |
| Pneumothorax | 0.872 |
| Consolidation | 0.793 |
| Edema | 0.882 |
| Emphysema | 0.921 |
| Fibrosis | 0.805 |
| Pleural Thickening | 0.782 |
| Hernia | 0.924 |
| **Mean AUC** | **0.832** |

---

## 🔬 XAI — Explainability

Generate GradCAM heatmaps for any input image:

```bash
python src/xai/gradcam.py \
  --image data/images/00000001_000.png \
  --checkpoint results/checkpoints/best.pth \
  --class Effusion
```

SHAP prototype explanations (coming in v2):

```bash
python src/xai/shap_explain.py --checkpoint results/checkpoints/best.pth
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

CI runs automatically on every PR via GitHub Actions.

---

## 📝 Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{modi2026medvision,
  author    = {Zeeshan Modi},
  title     = {MedVision AI: Multi-label Chest Pathology Classification with XAI and Fairness},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Zeesejo/medvision-ai}
}
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy, commit message format, and how to file issues.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
