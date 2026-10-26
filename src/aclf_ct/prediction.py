"""Test-cohort prediction using fitted models and fixed thresholds."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from .io import (read_table, selected_matrix, check_disjoint, new_output,
                 write_json, sha256, profile, probabilities)
from .classifiers import get_proba_1
from .evaluation import evaluate_vector


def predict_bundle(bundle, internal, external, out, trusted=False, bootstraps=2000):
    if not trusted:
        raise ValueError('Only load author-controlled bundles; pass --trust-model-files.')
    bundle = Path(bundle)
    info = json.loads((bundle / 'training.json').read_text(encoding='utf-8'))
    model_path = bundle / 'probability_model.joblib'
    if sha256(model_path) != info['probability_model_sha256']:
        raise ValueError('Probability model changed after training metadata was written.')
    membership = set(json.loads((bundle / 'training_membership.json').read_text(encoding='utf-8')))
    paths = dict(internal=internal, external=external)
    frames = {s: read_table(p) for s, p in paths.items()}
    check_disjoint(frames.values())
    matrices = {}
    paper = profile()[info['model']]
    for split, frame in frames.items():
        if 'cohort' in frame and not frame.cohort.eq(split).all():
            raise ValueError('Test-cohort role disagrees with the input cohort column.')
        hashes = {hashlib.sha256(str(x).encode('utf-8')).hexdigest() for x in frame.imageName}
        if hashes & membership:
            raise ValueError('Prediction input overlaps the fitting cohort.')
        X, y = selected_matrix(frame, info['features'])
        if info['mode'] == 'frozen_paper_features':
            if len(y) != paper['counts'][split] or int(y.sum()) != paper['events'][split]:
                raise ValueError('Test counts differ from the paper profile.')
        else:
            expected = info['expected_test_matrices'].get(split)
            if expected is None or expected['sha256'] != sha256(paths[split]):
                raise ValueError('Test matrix is not the matching training-transformed selection output.')
        matrices[split] = (X, y)
    output = new_output(out, [bundle, *paths.values()])
    model = joblib.load(model_path)
    results = []
    for split, (X, y) in matrices.items():
        p = probabilities(get_proba_1(model, X), len(y))
        seed = paper['auc_bootstrap_seeds'][split] if info['mode'] == 'frozen_paper_features' else 42
        results.append(dict(model=info['model'], cohort=split,
                            **evaluate_vector(y, p, info['threshold'], bootstraps, auc_seed=seed)))
        pd.DataFrame({'imageName': frames[split].imageName, 'y_true': y, 'p_aclf': p,
                      'prediction': (p >= info['threshold']).astype(int)}).to_csv(
                          output / f'{split}_predictions.csv', index=False)
    pd.DataFrame(results).to_csv(output / 'performance.csv', index=False)
    write_json(output / 'prediction.json', dict(
        mode=info['mode'], training_metadata_sha256=sha256(bundle / 'training.json'),
        inputs={s: sha256(p) for s, p in paths.items()}, threshold=info['threshold'],
        test_fitting=False, patient_level_outputs='Private patient-level outputs.'
    ))
    return pd.DataFrame(results)
