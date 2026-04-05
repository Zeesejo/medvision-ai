# Paper Outline — MedVision-AI

## Target Venues (pick one)

| Venue | Deadline | Format | Pages |
|-------|----------|--------|-------|
| MICCAI 2026 | ~Feb 2027 | LNCS | 8-10 |
| MIDL 2026 | ~Jan 2027 | PMLR | 8 |
| ISBI 2027 | ~Oct 2026 | IEEE | 4 |
| arXiv (preprint) | Anytime | Any | Any |

**Recommendation:** Submit to arXiv first (free, immediate visibility),
then target ISBI 2027 (shorter paper, good for first publication).

---

## Paper Structure

### 1. Abstract (150-200 words)
- [ ] Problem statement (chest X-ray multi-label classification)
- [ ] Method summary (ResNet-50 + ASL + Grad-CAM)
- [ ] Key result (mean AUC = X.XX)
- [ ] Contribution (explainability + public UI)

### 2. Introduction (~1 page)
- [ ] Clinical motivation
- [ ] Challenge: multi-label + class imbalance + black-box
- [ ] Our 4 contributions (listed as bullet points)

### 3. Related Work (~0.5 page)
- [ ] Chest X-ray classification (CheXNet, etc.)
- [ ] Class imbalance losses (Focal, ASL)
- [ ] Explainability (Grad-CAM)

### 4. Methodology (~2 pages)
- [ ] Dataset description (NIH ChestX-ray14)
- [ ] Model architecture (ResNet-50 + custom head)
- [ ] Loss function (ASL equation)
- [ ] Training protocol
- [ ] Grad-CAM formulation

### 5. Experiments (~2 pages)
- [ ] Per-class AUC table (compare to Wang et al. baseline)
- [ ] Grad-CAM figure (4-6 example images)
- [ ] Ablation: BCE vs Focal vs ASL loss comparison

### 6. Discussion (~0.5 page)
- [ ] Analysis of strong/weak classes
- [ ] Limitations (NLP label noise, no label correlation)
- [ ] Future work (ViT backbone, fairness)

### 7. Conclusion (~0.25 page)

---

## TODO After Training

- [ ] Fill in mean AUC in abstract and Table 1
- [ ] Run `python src/evaluate.py` → get per-class AUC
- [ ] Generate 4-6 Grad-CAM figures for Figure 1
- [ ] Run ablation (BCE vs Focal vs ASL) — retrain x2
- [ ] Screenshot Gradio UI for paper figure
- [ ] Write Discussion section based on actual results
- [ ] Proofread + submit to arXiv

---

## Key Novelty for Reviewers

1. **ASL loss** on ChestX-ray14 — not widely evaluated here
2. **Grad-CAM integration** with multi-label head — class-specific maps
3. **Clinical UI** — practical deployment contribution
4. **Reproducible** — full open-source code on GitHub
