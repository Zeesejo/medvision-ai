# Data

This directory contains dataset download scripts and preprocessing utilities.

## Available Datasets

### 1. NIH Chest X-Ray
- **Size:** ~112,000 images, 14 disease labels
- **Source:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Task:** Multi-label disease classification, fairness auditing, SSL pretraining

### 2. MedMNIST
- **Size:** Multiple small standardized medical datasets
- **Source:** https://medmnist.com/
- **Install:** `pip install medmnist`

### 3. Skin Lesion / Dermoscopy
- **Source:** https://www.isic-archive.com/
- **Task:** Benign vs. malignant classification, XAI evaluation

## Notes
- Raw data files are excluded from git (see `.gitignore`)
- Download scripts will be added in `data/scripts/`
