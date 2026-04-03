# Research Notes

## Project Summary

This project contributes to medical computer vision by addressing three open challenges identified in recent surveys (MICCAI 2024, PMC 2026):

1. **Label scarcity** — Medical annotations are expensive; self-supervised learning can reduce dependence on labels.
2. **Explainability** — Clinicians need interpretable predictions; prototype/attention methods go beyond post-hoc heatmaps.
3. **Fairness** — Models must perform equitably across demographic subgroups (age, sex, race/ethnicity where available).

## Key References

- NIH Chest X-Ray dataset: Wang et al., CVPR 2017
- SimCLR: Chen et al., ICML 2020
- Masked Autoencoders (MAE): He et al., CVPR 2022
- MoCo: He et al., CVPR 2020
- Grad-CAM: Selvaraju et al., ICCV 2017
- ProtoPNet: Chen et al., NeurIPS 2019
- Fairness in medical AI: Obermeyer et al., Science 2019

## Open Questions

- Which SSL method (contrastive vs. generative) generalizes better across medical modalities?
- Can prototype-based models match discriminative accuracy while improving explanation quality?
- How stable are fairness metrics under dataset shift?
