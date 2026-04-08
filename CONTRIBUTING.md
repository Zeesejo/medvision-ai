# Contributing to MedVision-AI

Thank you for your interest in contributing! This project focuses on self-supervised learning, XAI, and fairness in medical imaging.

## Getting Started

```bash
git clone https://github.com/Zeesejo/medvision-ai.git
cd medvision-ai
conda env create -f environment.yml
conda activate medvision
```

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only |
| `develop` | Integration branch |
| `feat/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `exp/<name>` | Experiment runs |
| `docs/<name>` | Documentation only |

**Never push directly to `main`.** Always open a PR.

## Workflow

1. Create a branch: `git checkout -b feat/your-feature`
2. Make focused, atomic commits
3. Open a PR against `main`
4. Request a review — all PRs require at least one approval
5. Merge only after CI passes

## Commit Message Format

```
<type>: <short description>

Types: feat | fix | exp | docs | refactor | chore | test

Examples:
  feat: add DenseNet-121 backbone support
  fix: normalize class weights in get_class_weights()
  exp: ablation study — focal loss vs ASL on ChestX-ray14
  docs: update README with dataset setup instructions
```

## Code Style

- Max line length: 120 characters
- Use type hints for all function signatures
- Docstrings on all public functions and classes
- Run `flake8 src/` before submitting

## Reporting Bugs

Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template. Include environment details and a minimal reproduction.

## Suggesting Features

Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template.

## Questions?

Open a GitHub Discussion or file an issue with the `question` label.
