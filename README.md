# MedVision-AI

**Reproducible research on multi-label chest X-ray classification**  
NIH ChestX-ray14 · PyTorch · DenseNet/ResNet · Asymmetric Loss · Explainability

> **Research status:** active, unpublished work. The repository contains completed baseline experiments as well as unfinished research prototypes. Only results explicitly listed as **completed** below should be treated as experimental findings.

## Why this repository exists

MedVision-AI studies multi-label chest pathology classification on the NIH ChestX-ray14 dataset. The project is being reorganized around a publication-quality workflow: reproducible training, preserved experiment provenance, controlled ablations, uncertainty reporting, explainability evaluation, and subgroup analysis.

The current repository should be understood in two layers:

1. **Completed historical baseline experiments** — ResNet-50 and DenseNet-121 test AUROC artifacts are committed.
2. **Research directions under development** — self-supervised learning, fairness auditing, ViT experiments, SHAP, and quantitative XAI are implemented or scaffolded to varying degrees but do **not** yet have complete committed result sets.

## Completed results

The strongest auditable result artifacts currently committed under `results/metrics/` are:

| Backbone | Mean AUROC | Status |
|---|---:|---|
| DenseNet-121 | **0.7978** | completed historical run |
| ResNet-50 | **0.7067** | completed historical run |

DenseNet-121 had higher recorded AUROC than ResNet-50 for all 14 pathologies in these artifacts.

> These are historical single-run artifacts, not a claim of state-of-the-art performance. The exact historical runtime environment, raw predictions, checkpoints, and all experiment logs were not preserved in Git. The current cleanup therefore treats these values as **baseline evidence to reproduce**, not as the final publication result.

Per-class metrics are available in:

```text
results/metrics/densenet121_auc.json
results/metrics/resnet50_auc.json
```

## Research status

| Component | Status |
|---|---|
| DenseNet-121 baseline | ✅ completed result artifact |
| ResNet-50 baseline | ✅ completed result artifact |
| ROC figures | ✅ committed |
| Asymmetric Loss implementation | ✅ implemented |
| BCE / Focal / ASL controlled ablation | 🟡 planned / needs rerun |
| ViT comparison | 🟡 config/code exists; no completed result artifact |
| SimCLR / SSL | 🟡 prototype; no completed NIH downstream result |
| Grad-CAM | 🟡 implementation exists; quantitative validation pending |
| SHAP | 🟡 prototype |
| Fairness by age/sex | 🟡 scaffold; real audit pending |
| Calibration analysis | 🔴 not yet part of the completed study |
| Multi-seed confidence intervals | 🔴 not yet part of the completed study |
| External validation | 🔴 not yet completed |

See [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) for the evidence/provenance audit.

## Repository layout

```text
medvision-ai/
├── config.yaml                 # current baseline configuration
├── configs/                    # experimental configurations
├── src/
│   ├── data/                   # dataset and loader code
│   ├── models/                 # classifier/backbone wrappers
│   ├── ssl_pretrain/           # SSL research code
│   ├── xai/                    # Grad-CAM / SHAP code
│   ├── losses.py
│   ├── train.py                # current canonical trainer candidate
│   └── evaluate.py
├── experiments/
│   ├── fairness/
│   ├── finetune/
│   ├── ssl/
│   └── xai/
├── results/
│   ├── metrics/                # committed historical metrics
│   └── figures/                # committed figures
├── reproducibility/            # experiment provenance templates
├── paper/                      # manuscript materials
├── tests/
└── docs/
```

## Dataset

This project uses the **NIH ChestX-ray14** dataset: 112,120 frontal chest radiographs with 14 automatically extracted pathology labels.

Expected metadata/split files include:

```text
Data_Entry_2017.csv
train_val_list.txt
test_list.txt
```

The official NIH train/test lists are used by the dataset loader. The internal validation strategy is being reviewed as part of the reproducibility cleanup; publication experiments should use an explicitly documented patient-disjoint protocol.

Dataset images are **not** distributed in this repository.

## Environment

Conda:

```bash
conda env create -f environment.yml
conda activate medvision
```

or install the Python dependencies from `requirements.txt` where applicable.

## Training

The current main trainer reads `config.yaml` by default and supports CLI overrides:

```bash
python src/train.py
python src/train.py --backbone densenet121
python src/train.py --backbone resnet50
python src/train.py --epochs 30 --lr 1e-4
python src/train.py --loss asl
python src/train.py --data_dir /path/to/nih-chestxray14
```

### Important reproducibility note

Multiple trainer generations (`train.py`, `train_v2.py`, `train_v3.py`) exist in the historical codebase and contain different training strategies. The publication cleanup will consolidate these into one canonical trainer before new headline results are reported.

## Evaluation

`src/evaluate.py` computes per-class ROC-AUC and the macro mean AUROC from a saved checkpoint. For future publication runs, evaluation will additionally preserve raw predictions and include AUPRC, calibration metrics, bootstrap confidence intervals, and subgroup metrics.

## Publication roadmap

The next experimental sequence is intentionally controlled:

1. make one canonical deterministic training/evaluation pipeline;
2. reproduce the DenseNet-121 and ResNet-50 historical baselines;
3. run BCE vs Focal vs ASL under identical conditions;
4. run multiple random seeds and bootstrap confidence intervals;
5. evaluate self-supervised pretraining under reduced label budgets;
6. add calibration and age/sex subgroup analysis;
7. quantitatively evaluate Grad-CAM/localization where annotations permit;
8. perform external validation if a compatible dataset/protocol is selected.

## Reproducibility policy for new experiments

Every new publication-facing run should preserve:

- Git commit SHA
- configuration file
- random seed
- dataset/split identifiers or hashes
- environment/package versions
- checkpoint identifier
- raw test probabilities and labels
- per-class metrics
- aggregate metrics
- training history
- generated figures

The template lives in [`reproducibility/`](reproducibility/).

## Paper

The `paper/` directory contains manuscript material. Until a submission/preprint is explicitly linked here, the manuscript and repository should be treated as **unpublished ongoing research**.

## License

MIT License. Dataset licensing and access terms are governed separately by the NIH ChestX-ray14 source.
