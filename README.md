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

---

## 🎯 Project Goal

Research, train, and deploy deep learning models for medical image analysis with a focus on:
- **Label efficiency** via self-supervised learning (SSL)
- **Explainability** via prototype/attention-based methods
- **Fairness** via demographic subgroup auditing
- **Deployment** via a clinical-grade web UI

Target output: a published paper at MICCAI, MIDL, or ISBI.

---

## 📁 Repository Structure

```
medvision-ai/
├── data/               # Dataset scripts and download utilities
├── experiments/        # Training scripts and experiment configs
├── models/             # Model architectures
├── notebooks/          # Exploratory analysis and result visualizations
├── ui/                 # Clinical web interface (React/plain HTML)
├── paper/              # LaTeX paper draft and figures
├── results/            # Saved checkpoints, logs, metrics
└── docs/               # Project documentation and research notes
```

---

## 🔬 Research Directions

| Direction | Status | Dataset |
|---|---|---|
| Self-supervised pretraining (SSL) | 🟡 Planning | NIH Chest X-Ray |
| Explainable AI (Prototype/XAI) | 🟡 Planning | NIH Chest X-Ray, Skin Lesion |
| Fairness & Bias Auditing | 🟡 Planning | NIH Chest X-Ray |
| Mobile/Smartphone Deployment | 🔴 Backlog | Wound / Skin Datasets |

---

## 🛠️ Tech Stack

- **Framework:** PyTorch + torchvision
- **Experiment tracking:** Weights & Biases (wandb)
- **UI:** Gradio / React
- **Paper:** LaTeX (Overleaf)
- **Environment:** conda (Python 3.10+)

---

## ⚙️ Setup

```bash
git clone https://github.com/Zeesejo/medvision-ai.git
cd medvision-ai
conda env create -f environment.yml
conda activate medvision
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
