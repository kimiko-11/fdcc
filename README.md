# Feature-Drift Guided Confidence Calibration Under Distribution Shift (FDCC)

## Project Description
FDCC is a final-year research project investigating confidence calibration under distribution shift by leveraging penultimate feature-space displacement signals in addition to output-space uncertainty.

## Research Question
Can penultimate feature-space drift predict the confidence correction needed under distribution shift, and does feature-space information provide complementary value beyond softmax entropy and related output-space signals?

## Hypotheses
- Feature-space displacement signals correlate with calibration error under shift.
- Feature-space signals improve confidence correction beyond output-only uncertainty measures.
- Adaptive temperature calibration informed by feature drift can outperform static calibration baselines on shifted data.

## Core Experimental Pipeline
1. Train/evaluate a CIFAR-10 baseline (ResNet-18).
2. Extract logits, probabilities, and penultimate features.
3. Evaluate shift behavior on CIFAR-10-C (and later held-out shifts/severities).
4. Compute uncertainty and feature-distance signals.
5. Compare calibration methods:
   - Uncalibrated
   - Temperature Scaling
   - Entropy-based calibration
   - Energy-based calibration
   - Feature-distance calibration
   - Proposed FDCC adaptive temperature calibration
6. Report robustness and generalization metrics with multi-seed statistics.

## Dataset and Model Scope
- **Primary dataset:** CIFAR-10
- **Shift benchmark:** CIFAR-10-C
- **Generalization checks:** held-out CIFAR-10-C settings, CIFAR-10.1 (optional real domain-shift dataset if time permits)
- **Model backbone:** ResNet-18

## Planned Baselines and Signals
- **Feature signals:** soft class-conditional Mahalanobis (`d_soft`), global Mahalanobis (`d_global`), minimum class distance (`d_min`)
- **Output signals:** softmax entropy, energy
- **Calibration baselines:** uncalibrated, TS, entropy-based, energy-based, feature-distance based, FDCC adaptive temperature

## Evaluation Metrics
- Accuracy
- ECE
- Adaptive/Quantile ECE
- NLL
- Brier Score
- Robustness analyses (5 seeds, mean ± 95% CI, permutation/null tests, norm vs angular, clean-data sanity, high-confidence error analysis)

## Repository Structure
```text
fdcc/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── configs/
│   └── baseline.yaml
├── src/
│   └── fdcc/
│       ├── __init__.py
│       ├── data/
│       │   └── __init__.py
│       ├── models/
│       │   └── __init__.py
│       ├── features/
│       │   └── __init__.py
│       ├── calibration/
│       │   └── __init__.py
│       ├── evaluation/
│       │   └── __init__.py
│       └── utils/
│           └── __init__.py
├── notebooks/
├── experiments/
├── results/
├── checkpoints/
└── docs/
```

## Current Project Status
Initial repository scaffold and placeholders are in place. No experiments, training runs, calibration implementations, or result generation have been completed yet.

## Reproducibility
- Planned experiments will use fixed seeds and report mean ± 95% confidence intervals across runs.
- Config-driven experiments will be version controlled.
- Heavy training/analysis will be run in Kaggle GPU notebooks, while code and experiment definitions are maintained in GitHub.

## Data and Artifact Policy
Large datasets, trained checkpoints, and generated experiment artifacts should **not** be committed to Git. Keep repository contents lightweight and reproducible via code/configuration.
