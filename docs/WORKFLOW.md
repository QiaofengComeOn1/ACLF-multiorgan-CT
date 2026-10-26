# Analysis workflow

Commands below use placeholder local paths. Each output directory must be new and separate from its inputs.

## 1. Fine-tune Inception-v3

```bash
aclf-ct fine-tune --manifest /inputs/liver_training_images.csv --training-cohort /inputs/prediction_training_ids.csv --weights /inputs/imagenet_inception_v3.pth --organ Liver --device cuda --out /results/liver_backbone
```

Repeat for Spleen and Muscle. Each manifest covers the prediction training cohort with matching patient identifiers and outcomes. Training uses an outcome-stratified 80:20 split, 20 head-training epochs at learning rate 0.001 and 80 full-network epochs at 0.0001, with Adam, batch size 16 and cross-entropy loss. The best second-stage validation-accuracy checkpoint is saved.

## 2. Extract organ embeddings

```bash
aclf-ct extract --manifest /inputs/liver_images.csv --weights /results/liver_backbone/inception_5ch.pth --organ Liver --device cuda --out /results/liver_embeddings
aclf-ct extract --manifest /inputs/spleen_images.csv --weights /inputs/spleen_checkpoint.pth --organ Spleen --device cuda --out /results/spleen_embeddings
aclf-ct extract --manifest /inputs/muscle_images.csv --weights /inputs/muscle_checkpoint.pth --organ Muscle --device cuda --out /results/muscle_embeddings
```

Each output contains metadata and 2,048 organ features. Training and extraction share masked bounding-box cropping, zero additional margin, resizing to 224 × 224 and per-input standardisation. The input manifests retain the study's train/internal/external allocation.

## 3. Fuse candidate features

```bash
aclf-ct fuse --liver /results/liver_embeddings/features.csv --spleen /results/spleen_embeddings/features.csv --muscle /results/muscle_embeddings/features.csv --clinical /inputs/clinical_candidates.csv --clinical-features-json /inputs/clinical_candidate_names.json --out /results/fusion
```

The candidate-name JSON identifies the clinical and quantitative CT predictors. The manuscript's candidate set comprises 15 clinical variables and four quantitative CT measurements. Fusion aligns patient identifiers, outcomes and cohort allocation across all tables.

## 4. Select features

```bash
aclf-ct select-features --training /results/fusion/CLSMFusion/train.csv --internal /results/fusion/CLSMFusion/internal.csv --external /results/fusion/CLSMFusion/external.csv --features-json /results/fusion/CLSMFusion/candidates.json --frequency 0.50 --out /results/selected
```

The procedure uses 30 stratified resampling iterations, Mann–Whitney screening with FDR control, Spearman filtering and L1 logistic selection. Cleaning is fitted within each training partition. Inner L1 cross-validation fits imputation and scaling within its training folds. The final training-fitted transforms are applied to both test cohorts.

Outputs include selection frequencies, selected parameters, the fitted cleaner/scaler, selected matrices and a separate L1 score. Features meeting the stability threshold remain in the exported matrices.

## 5. Fit and test the prediction model

```bash
aclf-ct train --training-csv /results/selected/train_scaled.csv --selection-dir /results/selected --model CLSMFusion --out /results/model
aclf-ct predict --bundle /results/model --internal /results/selected/internal_scaled.csv --external /results/selected/external_scaled.csv --trust-model-files --out /results/predictions
```

Training saves the raw estimator, probability estimator, feature order and training-derived threshold. Prediction uses this bundle and the associated transformed test matrices. Classifier comparison is conditional on the selected features. Patient-level prediction files are private outputs.

For existing selected matrices, omit `--selection-dir` and supply the features in `paper_profile.json`. The selected matrices must match the model feature order. `prepare-selected` accepts unscaled selected matrices and fits the final scaler using the training cohort.

## 6. Evaluate saved models or probabilities

```bash
aclf-ct evaluate --runs-root /inputs/profile_matched_runs --trust-model-files --out /results/model_evaluation
aclf-ct evaluate-probabilities --input /inputs/fixed_probabilities.csv --threshold 0.5 --out /results/probability_evaluation
```

The saved-run command requires compatible feature profiles, matrices and fitted models for all six configurations. For fixed probabilities, replace the example threshold with the model's fixed operating threshold.

## 7. Interpretation and exploratory analyses

```bash
aclf-ct attribute --run "/inputs/LSMFusion model" --cohort external --trust-model-files --out /results/feature_shap
aclf-ct attribute-images --config /inputs/image_attribution.json --device cuda --trust-model-files --out /results/image_attribution
aclf-ct cluster --training /inputs/training_aclf.csv --external /inputs/external_aclf.csv --out /results/clusters
aclf-ct organ-contributions --training /inputs/training_clusters.csv --out /results/organ_dependence
aclf-ct survival --training /inputs/training_clusters.csv --external /inputs/external_clusters.csv --out /results/survival
aclf-ct clinical-profiles --input /inputs/external_clusters.csv --out /results/clinical_profiles
aclf-ct meld --input /inputs/recorded_meld.csv --out /results/meld
```

Input schemas are listed in [INPUTS.md](INPUTS.md). Repository scope and analysis-version information are provided in the [README](../README.md).
