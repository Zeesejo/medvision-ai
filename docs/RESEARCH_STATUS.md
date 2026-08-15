# Research status and evidence audit

This document separates **completed evidence** from **implemented but unvalidated research directions** in MedVision-AI.

## Completed historical evidence

### DenseNet-121
- Mean AUROC: **0.7978**
- Per-class AUROC artifact: `results/metrics/densenet121_auc.json`
- ROC figure is committed under `results/figures/`.

### ResNet-50
- Mean AUROC: **0.7067**
- Per-class AUROC artifact: `results/metrics/resnet50_auc.json`
- Three recorded class AUROCs are exactly 0.5000: Atelectasis, Effusion, and Infiltration.

The cause of those chance-level ResNet values is **not established by the committed artifacts**. They should not be described as proven class collapse without a reproducible rerun and inspection of predictions/checkpoints.

## What is not currently reproducible from Git alone

The historical result files do not preserve all of the following together:

- exact runtime/package environment for each final run;
- exact trainer version used for each result artifact;
- raw test logits/probabilities;
- historical checkpoints;
- complete W&B logs/run identifiers;
- random seed provenance;
- repeated-run statistics.

For this reason, the historical AUROCs are treated as baseline artifacts to reproduce rather than final publication claims.

## Code paths that need consolidation

The repository contains multiple training generations:

- `src/train.py`
- `src/train_v2.py`
- `src/train_v3.py`

They do not represent one identical protocol. Before new publication-facing experiments, one canonical trainer should be selected and the others clearly marked as historical or migrated.

## Implemented/prototyped research that is not yet a completed result

### Self-supervised learning
`experiments/ssl/pretrain_simclr.py` and `src/ssl_pretrain/` contain SSL work, but no complete, auditable NIH ChestX-ray14 downstream SSL result set is currently committed.

### Fairness
`experiments/fairness/audit_subgroups.py` contains subgroup-audit logic/scaffolding, but the current committed repository does not contain a completed age/sex fairness result table from the target dataset.

### Explainability
Grad-CAM implementations exist under `src/xai/` and `experiments/xai/`. They support explanation generation, but the repository does not yet provide a quantitative localization study or expert validation that would support claims about clinical/anatomical explanation quality.

### Vision Transformer
ViT configuration/support exists, but no completed ViT metric artifact comparable to the DenseNet-121 and ResNet-50 JSON files is committed.

## Publication rule

A result should enter the abstract/main results table only when its provenance is recoverable from a run manifest containing at minimum:

1. commit SHA;
2. config;
3. seed;
4. split identifier/hash;
5. environment versions;
6. checkpoint/run ID;
7. raw predictions;
8. metric output;
9. figure/table generation path.

## Next reproducibility milestone

The immediate target is a one-command baseline run that trains and evaluates DenseNet-121 end-to-end and saves all publication-facing artifacts. Once that is stable, the same protocol should be applied to ResNet-50 before adding new ablations.