# MedVision-AI

**Reproducible research on multi-label chest X-ray classification**  
NIH ChestX-ray14 · PyTorch · DenseNet/ResNet · Asymmetric Loss · Explainability

> **Research status:** active, unpublished work. The repository contains completed historical baseline artifacts and research components that are still being validated. Only results explicitly marked as completed below should be treated as experimental findings.

## Research objective

MedVision-AI studies multi-label chest pathology classification on NIH ChestX-ray14 with an emphasis on **reproducibility, label efficiency, class imbalance, calibration, explainability, and subgroup robustness**.

The repository is currently being converted from an exploratory research codebase into a publication-grade experiment pipeline. Historical artifacts are preserved rather than silently rewritten, while new runs follow a stricter provenance policy.

## Completed historical results

The strongest auditable metric artifacts currently committed under `results/metrics/` are:

| Backbone | Mean AUROC | Status |
|---|---:|---|
| DenseNet-121 | **0.7978** | completed historical run |
| ResNet-50 | **0.7067** | completed historical run |

DenseNet-121 had higher recorded AUROC than ResNet-50 for all 14 pathologies in those artifacts.

These are **historical single-run results**, not state-of-the-art claims. Raw predictions, the exact runtime environment, and every historical checkpoint were not preserved in Git, so these values are treated as baseline evidence to reproduce.

```text
results/metrics/densenet121_auc.json
results/metrics/resnet50_auc.json
```

## What is completed vs planned

| Component | Status |
|---|---|
| DenseNet-121 historical baseline | completed result artifact |
| ResNet-50 historical baseline | completed result artifact |
| Historical ROC figures | committed |
| Asymmetric Loss | implemented |
| Canonical reproducible trainer | implemented on cleanup branch |
| Patient-disjoint validation split generation | implemented on cleanup branch |
| BCE / Focal / ASL controlled ablation | planned |
| ViT comparison | code/config exists; no completed result artifact |
| SimCLR / SSL | prototype; no completed NIH downstream result |
| Grad-CAM | implementation exists; quantitative validation pending |
| Fairness by age/sex | scaffold; real audit pending |
| Calibration analysis | planned |
| Multi-seed confidence intervals | planned |
| External validation | not yet completed |

See [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) for the evidence and provenance audit.

## Repository structure

```text
medvision-ai/
├── configs/
│   └── baseline_densenet121_asl.yaml   # canonical baseline experiment
├── scripts/
│   └── prepare_csv.py                  # deterministic split generation
├── src/
│   ├── constants.py                    # canonical class definitions
│   ├── dataset.py                      # canonical CSV-backed dataset
│   ├── models/
│   ├── losses.py
│   ├── reproducibility.py              # seed + environment/provenance utilities
│   ├── train.py                        # canonical train + test runner
│   ├── xai/
│   └── ssl_pretrain/
├── experiments/                        # research prototypes / older experiment code
├── results/
│   ├── metrics/                        # committed historical metrics
│   ├── figures/                        # committed historical figures
│   └── runs/                           # generated publication-facing runs; gitignored
├── reproducibility/
├── paper/
├── tests/
└── docs/
```

## Dataset

The project uses **NIH ChestX-ray14**, containing 112,120 frontal chest radiographs with 14 automatically extracted pathology labels.

Expected source files include:

```text
Data_Entry_2017.csv
train_val_list.txt
test_list.txt
images_001/ ... images_012/
```

Dataset images are not distributed in this repository.

### Prepare reproducible splits

The official NIH train/test lists are preserved. The publication pipeline creates the internal validation set at the **patient level** when `Patient ID` is available, preventing train/validation patient overlap.

```bash
python -m scripts.prepare_csv \
  --archive_dir /path/to/nih-chestxray14 \
  --output_dir data/splits \
  --strategy patient \
  --val_frac 0.10 \
  --seed 42
```

The command writes:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
data/splits/split_manifest.json
```

`split_manifest.json` records split counts, strategy, seed, and SHA-256 hashes of the generated CSV files.

For historical reproduction investigations, `--strategy image` is retained as an explicit alternative. Publication-facing results should not mix the two protocols.

## Environment

```bash
conda env create -f environment.yml
conda activate medvision
```

## Canonical baseline run

The publication workflow now uses one config-driven runner:

```bash
python -m src.train --config configs/baseline_densenet121_asl.yaml
```

The named baseline config currently specifies DenseNet-121, 320 px inputs, ImageNet initialization, AdamW, Asymmetric Loss, deterministic seed 42, early stopping, and cosine learning-rate scheduling.

Every run creates a dedicated directory such as:

```text
results/runs/E01_densenet121_asl_seed42/
├── config.yaml
├── environment.json
├── run_manifest.json
├── training_history.csv
├── best.pth
├── best_validation_metrics.json
├── test_metrics.json
└── test_predictions.csv
```

The important change is that **raw test probabilities and labels are preserved**. Future confidence intervals, calibration, subgroup analysis, ROC/PR figures, and paired model comparisons can therefore be regenerated without retraining the network.

## Historical trainers

`train_v2.py` and `train_v3.py` are retained for provenance because they represent earlier generations of the research code and use different training strategies. They should be treated as **legacy historical implementations**, not as competing publication entry points.

New publication-facing experiments should use `src/train.py` plus a named YAML config under `configs/`.

## Publication experiment roadmap

The experimental sequence is intentionally controlled:

1. reproduce DenseNet-121 and ResNet-50 baselines with the canonical pipeline;
2. run BCE vs Focal vs ASL under otherwise identical conditions;
3. repeat key experiments across multiple fixed seeds;
4. compute bootstrap confidence intervals and paired comparisons from stored predictions;
5. evaluate self-supervised pretraining under reduced label budgets;
6. add calibration and age/sex subgroup analysis;
7. quantitatively evaluate explanation localization where annotations permit;
8. perform external validation under a clearly documented label mapping and protocol.

## Reproducibility policy

Every new publication-facing result should preserve:

- Git commit SHA
- resolved experiment configuration
- random seed
- split strategy and split hashes
- package / runtime environment
- checkpoint
- training history
- raw labels and predicted probabilities
- per-class AUROC and AUPRC
- aggregate metrics
- generated figures and statistical analyses

## Paper

The `paper/` directory contains manuscript material. Until a submission or preprint is explicitly linked here, it should be treated as **ongoing unpublished research**.

## License

MIT License. NIH ChestX-ray14 licensing and access terms are governed separately by the dataset provider.
