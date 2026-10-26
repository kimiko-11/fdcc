# FDCC Experimental Protocol v1

**Protocol frozen:** 2026-10-26T12:05:23+05:30 (Asia/Kolkata)  
**Status:** Frozen before any new experiment under this protocol  
**Scope:** Feature-Drift Guided Confidence Calibration Under Distribution Shift  
**Primary benchmark:** CIFAR-10-C  
**Primary comparison:** entropy-only calibration versus entropy plus global feature-distance calibration

This document defines the planned splits, estimators, baselines, primary endpoint, and statistical tests. Do not change these after inspecting results from the frozen evaluation split. If a change becomes necessary, create a dated protocol revision and mark affected results as exploratory; do not silently overwrite this version.

## 1. Research question and hypotheses

**Research question:** Does penultimate feature-space displacement predict the confidence correction needed under distribution shift, and does it provide useful information beyond output-space uncertainty?

- **H1:** Feature-space displacement is associated with prediction error and calibration degradation under shift.
- **H2 (primary):** Adding a feature-distance signal to entropy-based temperature calibration improves held-out negative log-likelihood (NLL) on unseen corruption types.
- **H3:** A bounded, lightweight adaptive temperature informed by feature distance can improve calibration without target-evaluation labels, classifier retraining on target data, or test-time backpropagation.

The confirmatory claim is limited to the predeclared primary comparison and endpoint. Other metrics, signals, corruption subsets, and analyses are secondary or exploratory.

## 2. Data splits and leakage controls

### 2.1 CIFAR-10 training data

- Split the official 50,000-image CIFAR-10 training set into **45,000 source-training images** and **5,000 clean-validation images**.
- Make the split stratified by class, deterministic, and shared across all model seeds. Use split seed **20261026** and save the exact image indices in a version-controlled split manifest before training.
- Train each ResNet-18 seed only on the 45,000 source-training images.
- Use the 5,000 clean-validation images for the clean-validation temperature-scaling baseline and for the predeclared feature-statistics sensitivity analysis. Do not use them to train the classifier.

### 2.2 CIFAR-10-C image IDs

Each CIFAR-10-C corruption and severity is aligned to the 10,000 base CIFAR-10 test images. Split **base image IDs**, not individual corrupted files, into two disjoint, class-stratified sets:

- **Dev-images:** 5,000 base image IDs.
- **Eval-images:** the other 5,000 base image IDs.

Use a fixed split seed (**20261026**), publish the exact ID lists in the split manifest, and verify class balance. Do **not** use the first 5,000 versus last 5,000 IDs because CIFAR-10 test ordering can confound such a split with class identity. An image ID must never occur in both sets, even under different corruption types or severities.

Dev-images may be used to fit calibration mappings and select hyperparameters under the rules below. Eval-images are for locked evaluation only. Do not use Eval-image labels or metrics to choose a model, signal, mapping, hyperparameter, or stopping rule.

### 2.3 Corruption-type split

The 15 standard CIFAR-10-C corruption types are partitioned before new experiments:

| Role | Corruption types |
|---|---|
| Development types | gaussian_noise, glass_blur, motion_blur, contrast, pixelate |
| Unseen-type evaluation | shot_noise, impulse_noise, defocus_blur, zoom_blur, snow, frost, fog, brightness, elastic_transform, jpeg_compression |
| Final holdout | The four CIFAR-10-C extra corruptions: speckle_noise, gaussian_blur, spatter, saturate; plus CIFAR-10.1 as a separate secondary check if available |

Previously explored Gaussian-noise, glass-blur, and motion-blur results are treated as **exploratory development evidence**, not untouched confirmatory results. The new protocol and evaluation metrics must be frozen before running further experiments. Do not inspect final-holdout metrics until the core pipeline, all choices, and reporting code are locked.

### 2.4 Severity split

- Fit development calibrators using development types at severities **1, 3, and 5**.
- Evaluate unseen-severity generalization on development types at severities **2 and 4**, using Eval-images. This is a secondary generalization analysis and must not be used to tune the method.
- Evaluate unseen-type generalization on the ten unseen types at **all five severities**, using Eval-images.
- Evaluate the four extra corruption types and CIFAR-10.1 only after the core pipeline and reporting code are frozen. Do not use final-holdout results to revise the method and then report those same results as confirmatory.

## 3. Model and feature extraction

- Backbone: CIFAR-adapted ResNet-18, with a 3×3 first convolution, no initial max-pooling, and a 10-class final layer.
- Train **five seeds** with identical data splits. Predeclare seeds as **[11, 22, 33, 44, 55]**.
- The feature vector is the **512-dimensional penultimate representation after global average pooling and immediately before the final fully connected classifier**.
- Save raw logits, predicted classes, probabilities, 512-D features, labels, corruption type, severity, base image ID, and model seed for each evaluated sample.
- Fit each calibration method to the same logits for a given model seed. A strictly positive scalar temperature preserves the argmax for fixed logits; accuracy is reported for completeness, not as evidence of calibration improvement.

## 4. Feature-distance estimators

The primary feature statistics are fitted on representations extracted from the 45,000 source-training images. Use a tied covariance with Ledoit-Wolf shrinkage for the primary analysis. Document numerical regularization and use the same fitted source statistics for all evaluation examples of a given model seed.

Define all distances as **squared Mahalanobis distances**:

- **Primary feature signal — d_global:** squared Mahalanobis distance from the feature to the global source-feature mean.
- **Secondary feature signal — d_min:** minimum squared Mahalanobis distance to any source class mean, using the tied covariance.
- **Secondary signal — d_soft:** probability-weighted mean of the class-conditional squared Mahalanobis distances, where weights are the model's **T = 1 softmax probabilities**. The weights are fixed and do not depend on any learned temperature.

d_global is the primary distance for the confirmatory entropy-complementarity test because it does not use the classifier's predicted class or probabilities. Report d_min and d_soft separately as secondary analyses; do not select the best-performing distance after seeing Eval results.

Predeclared sensitivity analyses:
1. Fit feature statistics on the 5,000 clean-validation representations instead of source-training representations.
2. Compare tied covariance with per-class covariance, using shrinkage in both cases.
3. Compare feature-norm and angular/directional variants if implementation and time permit.

These are sensitivity analyses, not replacements for the primary estimator.

## 5. Calibration methods and fair comparisons

Report the following for each seed:

1. Uncalibrated model (T = 1).
2. **Clean-validation TS:** one scalar temperature fitted on the 5,000 clean-validation images.
3. **Development-shift TS:** one scalar temperature fitted on Dev-images from the five development corruption types at severities 1, 3, and 5.
4. Entropy-only adaptive calibration, T = g(H).
5. Energy-only adaptive calibration, T = g(E).
6. Distance-only adaptive calibration, T = g(d_global).
7. **Proposed primary method:** entropy plus global feature distance, T = g(H, d_global).
8. Secondary variants using d_min and d_soft.

For the fair development-fitted comparison (items 3–8), fit every method using the same development examples and labels. Clean-validation TS is a separate source-only reference and must not be presented as a matched-development-data comparison.

All fitted temperatures are bounded to **[0.5, 5.0]**. Use the same predeclared candidate mapping families and regularization search for all adaptive methods, with feature inputs standardized using development data only. Select the mapping family and regularization using the cross-validation procedure below. Minimize NLL on fit data. Do not use evaluation labels for fitting, hyperparameter selection, or early stopping.

## 6. Hyperparameter selection without image-ID leakage

Use leave-one-development-corruption-type-out cross-validation over the five development types. For each fold:

1. Hold out one corruption type for validation.
2. Within Dev-images, split base image IDs into disjoint, class-stratified fold-fit and fold-validation subsets (80% / 20% using a fixed, recorded seed).
3. Fit the calibration mapping on the other four development types, severities 1, 3, and 5, using only fold-fit IDs.
4. Validate on the held-out development corruption type at severities 1, 3, and 5 using only fold-validation IDs.
5. Aggregate validation NLL across the five held-out types to select the shared mapping family and regularization settings.

This prevents the same base image from appearing in both fit and validation within a fold. After selection, refit each development-fitted calibration method on all Dev-images and the five development types at severities 1, 3, and 5. Freeze the resulting method and parameters before evaluating unseen types.

Do not select a feature-distance estimator, corruption split, temperature bound, metric, or new hyperparameter based on Eval results.

## 7. Endpoints and reporting

### Primary endpoint

For the ten unseen corruption types, all five severities, and Eval-images, calculate the paired NLL improvement:

**NLL gain = NLL(entropy-only g(H)) − NLL(proposed g(H, d_global)).**

Positive values favor the proposed method. Pooling rules and sample weighting must be identical for both methods. Report per-corruption and per-severity results as well as the predeclared aggregate.

### Secondary outcomes

- ECE using a fixed, documented binning scheme; Adaptive/Quantile ECE.
- Brier score.
- Accuracy (expected to remain unchanged for positive temperature scaling applied to fixed logits).
- Entropy-only versus distance-only, energy-only, development-shift TS, and clean-validation TS comparisons.
- d_min and d_soft variants; feature-statistics sensitivity analyses.
- Error and calibration analyses for high-confidence incorrect predictions.

Use the same metric implementation and binning convention across methods. State the number of bins and empty-bin handling. ECE is secondary; do not use it to replace the primary endpoint after results are seen.

## 8. Uncertainty and seed reporting

- Run five model-training seeds with identical data splits.
- Report each seed's result and the mean across seeds. For headline tables, report the mean and a 95% t-interval over the five **paired seed-level differences**.
- Also compute a two-way cluster bootstrap over unseen corruption types and base image IDs, keeping all severities for a sampled corruption/image pair together. Report the bootstrap interval for the paired NLL gain and document the number of replicates and random seed.
- Present seed variability and bootstrap uncertainty distinctly; do not treat individual corrupted variants of the same base image as independent observations.

Five seeds provide a limited estimate of training variability. Avoid claiming broad statistical certainty from the seed count alone.

## 9. Conditional permutation null

Run 1,000 permutations as a secondary test of whether the sample-level d_global assignment adds information beyond entropy and coarse evaluation strata.

- Keep logits, entropy, labels, predictions, and the already-fitted calibrators fixed.
- Within the Eval set, permute d_global among samples within strata defined by corruption type, severity, predicted class, and entropy decile. Merge adjacent entropy bins deterministically where strata are too small, and record the rule.
- Recompute the held-out NLL gain for each permutation using the frozen calibrators; do not refit using Eval labels.
- Report the null distribution, observed gain, and the predeclared tail probability calculation.

This is a conditional diagnostic, not a substitute for the primary held-out comparison.

## 10. Local required-temperature analysis

Treat group-level required-temperature (T*) analysis as descriptive/secondary, not as a source of method-selection decisions.

- Define feature-distance bin edges from Dev data only and freeze them before evaluation.
- Estimate a single optimal temperature per sufficiently populated group by minimizing that group's NLL, without feeding those Eval-label-derived estimates back into g.
- Require approximately 500 samples per reported bin; merge adjacent bins according to a predeclared rule if needed.
- Report sample counts and uncertainty, and avoid interpreting noisy per-sample optimal temperatures as ground truth.

## 11. Reproducibility and stop rules

Before any new experiment:
1. Commit this protocol and the exact split-generation code/manifest.
2. Verify CIFAR-10-C array ordering and label alignment against the official CIFAR-10 test IDs.
3. Verify that source-train, clean-validation, Dev-images, and Eval-images have no unintended overlap.
4. Run only pipeline-integrity checks before the protocol and manifests are frozen; do not use Eval metrics to debug or tune model choices.
5. Record code commit, dataset source/version, library versions, model seed, split seed, and artifact hashes for every run.

If the protocol must change, create a new dated version, explain the reason and affected hypotheses/endpoints, and mark results generated under the changed design as exploratory unless the evaluation holdout remains genuinely untouched.

## 12. Timestamp and prior exploration

**Frozen at:** 2026-10-26T12:05:23+05:30 (Asia/Kolkata).  
The Git commit timestamp is the repository's authoritative record of when this protocol entered version control.

Previously examined results are exploratory. In particular, prior Gaussian-noise severity-1, glass-blur, and motion-blur summaries must not be presented as confirmatory evidence. The unseen-type evaluation is held out from the new calibration-fitting workflow, but the project should transparently disclose all prior aggregate analyses in reports.
