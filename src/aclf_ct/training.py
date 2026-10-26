"""Classifier fitting and cross-validation on selected training features."""
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from .io import profile, read_table, selected_matrix, new_output, write_json, sha256
from .classifiers import build_models, get_proba_1
from .metrics import calc_metrics, find_best_threshold_by_youden


def fit_model(training_csv, model_name, out, benchmark=False, selection_dir=None):
    """Fit and compare classifiers using fixed selected training features."""
    config = profile()[model_name].copy()
    selection = None
    if selection_dir is not None:
        selection_path = Path(selection_dir) / 'selection.json'
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        if sha256(training_csv) != selection['matrices']['train']['sha256']:
            raise ValueError('Training matrix does not match the supplied selection bundle.')
        config['features'] = selection['features']
    frame = read_table(training_csv)
    X, y = selected_matrix(frame, config['features'])
    if selection is None and (len(y) != config['counts']['train'] or int(y.sum()) != config['events']['train']):
        raise ValueError('Expected 394 training patients and 128 events.')
    if 'cohort' in frame and not frame.cohort.eq('train').all():
        raise ValueError('Only prediction-training cohort patients may enter model fitting.')
    output = new_output(out, [training_csv] + ([selection_dir] if selection_dir else []))
    candidates = build_models(None if benchmark else [config['classifier']])
    if not benchmark:
        candidates = {config['classifier']: candidates[config['classifier']]}
    fold_rows, failures = [], {}
    for name, estimator in candidates.items():
        try:
            for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X,y),1):
                fitted = clone(estimator).fit(X[tr], y[tr])
                p = get_proba_1(fitted, X[va])
                fold_rows.append(dict(
                    model=name,
                    fold=fold,
                    **calc_metrics(y[va], (p>=.5).astype(int),p)
                ))
        except Exception as exc:
            failures[name] = f'{type(exc).__name__}: {exc}'
    pd.DataFrame(fold_rows).to_csv(output/'cv_folds.csv', index=False)
    write_json(output/'failed_models.json', failures)
    if failures:
        raise RuntimeError('Classifier comparison failed; see candidate failure records.')
    summary = pd.DataFrame(fold_rows).groupby('model', sort=True).mean(numeric_only=True).drop(columns='fold').reset_index()
    summary = summary.sort_values(
        ['AUC','F1','Accuracy','Brier','model'],
        ascending=[False,False,False,True,True]
    )
    summary.to_csv(output/'leaderboard.csv', index=False)
    selected = str(summary.iloc[0]['model'])
    raw_model = clone(candidates[selected]).fit(X,y)
    joblib.dump(raw_model, output/'raw_model.joblib')
    if config['use_calibrated_probabilities']:
        probability_model = CalibratedClassifierCV(estimator=clone(raw_model), method='sigmoid', cv=5).fit(X,y)
    else:
        
        probability_model = raw_model
    joblib.dump(probability_model, output/'probability_model.joblib')
    
    write_json(output/'training_membership.json', sorted(
        hashlib.sha256(str(x).encode('utf-8')).hexdigest() for x in frame.imageName
    ))
    threshold, _ = find_best_threshold_by_youden(y, get_proba_1(probability_model,X))
    write_json(output/'training.json', dict(model=model_name, classifier=selected, features=config['features'],
        training_sha256=sha256(training_csv), training_n=len(y), training_events=int(y.sum()),
        threshold=threshold, calibrated=config['use_calibrated_probabilities'],
        matches_saved_classifier=selected == config['classifier'],
        mode='new_selection' if selection else 'frozen_paper_features',
        expected_test_matrices=selection['matrices'] if selection else None,
        probability_model_sha256=sha256(output/'probability_model.joblib'),
        validation_scope='Cross-validation conditional on fixed selected training features.'))
    return summary
