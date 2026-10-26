# Input specifications

Patient identifiers are study pseudonyms. Input files are stored outside the source repository. Each analysis requires a new output directory outside its input directories.

## Prediction tables and models

Feature matrices contain `imageName`, binary `Group` and the ordered predictors required by the model. Cohort labels are `train`, `internal` and `external`. The configured cohort sizes are 394, 161 and 159, with 128, 53 and 72 ACLF events, respectively.

| Configuration | Current feature count |
|---|---|
| ClinicalMP | 6 |
| Liver | 6 |
| Spleen | 13 |
| Muscle | 7 |
| LSMFusion | 13 |
| CLSMFusion | 21 |

Feature order is defined in `src/aclf_ct/paper_profile.json`. Fitted models and input matrices must use the same predictors in the same order. Analysis-version information is provided in the [README](../README.md).

The saved-run interface reads these files within each model directory:

```text
preprocess/train_scaled.csv
preprocess/internal_scaled.csv
preprocess/external_scaled.csv
train/leaderboard.json
train/best_raw_model_<classifier>.pkl
train/best_calibrated_model.pkl
```

The calibrated file is required for calibrated configurations. Matrix feature order and preprocessing must match the fitted model. Model loading requires `--trust-model-files` and trusted local artifacts.

## Feature selection

Candidate tables contain `imageName`, `Group`, optional `cohort`, and the predictors listed in the candidate-name JSON array. Candidate variables must be available at the prediction time. Missing numeric values are handled by training-fitted imputation.

## Fixed probabilities

Required columns: `y_true`, `p_aclf`. The classification threshold is supplied separately. Exact bootstrap comparison requires the original row order. Paired model comparisons require matching patient identifiers and outcomes.

## Clustering and organ dependence

Training and external ACLF tables contain 128 and 72 cases, respectively. Required fields are `imageName`, 2,048 `Liver_f_*` features, 2,048 `Spleen_f_*` features and 2,048 `Muscle_f_*` features. Primary clustering uses the concatenated 6,144-dimensional representation. If `Group` is supplied, every value is 1.

Training `time_days` and `event` are used for mortality-based display numbering. Organ-dependence analysis uses the training table with fixed `cluster` labels and deep-feature columns only. Single-organ PCA retains at most 50 components; leave-one-organ-out PCA uses a 90% explained-variance target.

## Survival

Required columns: `imageName`, `cluster`, `time_days`, `event`. Event coding is 1 for death and 0 for censoring. Follow-up time uses the study-defined time origin. Unavailable follow-up remains missing; analyses require both time and event status. Death on the endpoint day counts as an event. Zero follow-up times use the recorded lower bound of 1e-6 days, with affected counts reported.

## Packed organ images

Manifest columns: `imageName,image,mask,Group,cohort`. Image and mask paths may be absolute or relative to the manifest. Each pair has shape H × W × 5 and matching NIfTI geometry. CT resampling, slice selection and segmentation are upstream steps.

Training and extraction use the shared mask bounding box, zero additional margin, 224 × 224 resizing and per-input standardisation. Feature extraction requires five-channel Inception-v3 weights and exports 2,048 features per organ.

## Fusion image attribution

The configuration JSON contains `fusion_model`, `preprocessor`, `weights`, `images` and `background`. The final three fields each contain `Liver`, `Spleen` and `Muscle` entries. Each organ image has `image` and `mask` paths; each background entry is a non-empty list of such pairs. Relative paths are resolved against the configuration file.

Preprocessing may be a saved selection bundle containing `scaler` and `features`, or a saved scaler with an explicit ordered `features` array in the JSON. The fitted predictor is binary logistic regression, optionally with fitted sigmoid calibration. Other-organ inputs remain fixed during each organ's attribution.

## MELD

Required columns: `cohort`, `Group`, `MELD`. Cohort coding is 1, 2 and 3 for training, internal and external cohorts. The command evaluates recorded MELD scores. Missing score/outcome pairs are counted and excluded.
