"""Training-partition screening, stability selection and feature export."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from .io import read_table, binary_labels, check_disjoint, new_output, write_json, sha256
from .fusion import validate_predictor_names
from .selection_core import (fit_clean_transform_train, transform_other_set,
                             select_features_one_fold, logistic_l1_cv_select)


def historical_parameters():
    return dict(fdr_alpha=.05, corr_threshold=.9, Cs=np.logspace(-4, 2, 40),
                logit_cv_folds=10, random_state=42, logit_max_iter=20000,
                logit_n_jobs=8, logit_class_weight='balanced', use_C_rule='1se')


def final_stable_fit(X, y, features, params):
    """Fit the final L1 model on stability-selected features."""
    return logistic_l1_cv_select(
        X_train=X[features], y_train=y, Cs=params['Cs'], cv_folds=params['logit_cv_folds'],
        random_state=params['random_state'], max_iter=params['logit_max_iter'],
        n_jobs=params['logit_n_jobs'], class_weight=params['logit_class_weight'],
        use_C_rule=params['use_C_rule']
    )


def export_selected_matrices(frames, candidates, features, cleaner, final_fit, output):
    """Apply the fitted cleaner and scaler to each cohort."""
    records = {}
    for split, frame in frames.items():
        if not set(candidates).issubset(frame):
            raise ValueError(f'{split}: required candidate columns are missing.')
        values = frame[candidates].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
        cleaned = transform_other_set(values, cleaner)
        scaled = final_fit['scaler'].transform(cleaned[features])
        if not np.isfinite(scaled).all():
            raise ValueError('Nonfinite transformed features.')
        metadata = [c for c in ['imageName', 'Group', 'cohort'] if c in frame]
        table = pd.concat([frame[metadata].reset_index(drop=True),
                           pd.DataFrame(scaled, columns=features)], axis=1)
        path = output / f'{split}_scaled.csv'
        table.to_csv(path, index=False)
        score = frame[metadata].reset_index(drop=True).copy()
        score['Rad_score'] = final_fit['model_final'].decision_function(scaled)
        score.to_csv(output / f'{split}_selection_score.csv', index=False)
        records[split] = dict(n=len(frame), events=int(frame.Group.sum()), sha256=sha256(path))
    return records


def run_selection(training, candidate_names, frequency, out, internal=None, external=None):
    validate_predictor_names(candidate_names)
    if frequency != .5:
        raise ValueError('The stability-selection frequency must be 0.50.')
    if (internal is None) != (external is None):
        raise ValueError('Supply both test cohorts or neither.')
    paths = dict(train=training)
    if internal is not None:
        paths.update(internal=internal, external=external)
    frames = {split: read_table(path) for split, path in paths.items()}
    check_disjoint(frames.values())
    for split, frame in frames.items():
        binary_labels(frame.Group)
        if 'cohort' in frame and not frame.cohort.eq(split).all():
            raise ValueError('Cohort labels disagree with the supplied input role.')
        if not set(candidate_names).issubset(frame):
            raise ValueError(f'{split}: candidate predictors are incomplete.')
    frame = frames['train']
    y = binary_labels(frame.Group)
    X = frame[candidate_names].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    params = historical_parameters()
    output = new_output(out, paths.values())
    counts = pd.Series(0, index=X.columns, dtype=int)
    for tr, val in RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=42).split(X, y):
        chosen = select_features_one_fold(X.iloc[tr], y[tr], params, X.iloc[val])
        counts.loc[chosen] += 1
    frequencies = (counts / 30).sort_values(ascending=False)
    chosen = frequencies[frequencies >= frequency].index.tolist()
    if not chosen:
        raise RuntimeError('No features meet the selection criteria.')
    pd.DataFrame({'Feature': frequencies.index, 'Select_Frequency': frequencies.values,
                  'Select_Count': counts.loc[frequencies.index].values, 'Total_Folds': 30}).to_csv(
                      output / 'frequencies.csv', index=False)
    _, cleaner = fit_clean_transform_train(X)
    final = final_stable_fit(X, y, chosen, params)
    records = export_selected_matrices(frames, candidate_names, chosen, cleaner, final, output)
    joblib.dump(dict(cleaner=cleaner, scaler=final['scaler'], candidates=candidate_names,
                     features=chosen), output / 'preprocessor.joblib')
    joblib.dump(final['model_final'], output / 'selection_l1_model.joblib')
    pd.DataFrame({'Feature': chosen, 'coefficient': final['coef_series'].loc[chosen].values,
                  'mean_train': final['scaler'].mean_, 'scale_train': final['scaler'].scale_}).to_csv(
                      output / 'selected_parameters.csv', index=False)
    write_json(output / 'selection.json', dict(
        features=chosen, threshold=frequency, training_sha256=sha256(training), matrices=records,
        C_best=final['C_best'], C_1se=final['C_1se'], C_used=final['C_used'],
        intercept=final['intercept'], final_nonzero_features=final['selected_features'],
        scope='Repeated stratified screening and stability selection.',
        cleaning_scope='Training-partition cleaning; inner-fold imputation and scaling; final training-only refit.',
        C_rule='One foldwise standard deviation.',
        threshold_evaluation='Youden threshold fitted to training-cohort predictions.'
    ))
    return dict(selected_features=len(chosen), threshold=frequency, matrices=list(records))
