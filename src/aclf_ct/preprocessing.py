"""Standardise already selected features with training-only parameters."""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .io import (profile, read_table, check_disjoint, selected_matrix,
                 new_output, write_json, sha256)


def prepare_selected(training, internal, external, model_name, out):
    """Fit a training-cohort scaler to unscaled selected features and transform all cohorts."""
    paths = dict(train=training, internal=internal, external=external)
    frames = {s: read_table(p) for s, p in paths.items()}
    check_disjoint(frames.values())
    config = profile()[model_name]
    matrices = {}
    for split, frame in frames.items():
        if 'cohort' in frame and not frame.cohort.eq(split).all():
            raise ValueError('Input cohort assignment differs from its supplied role.')
        X, y = selected_matrix(frame, config['features'])
        if len(y) != config['counts'][split] or int(y.sum()) != config['events'][split]:
            raise ValueError('Cohort counts differ from the configured profile.')
        matrices[split] = X
    scaler = StandardScaler().fit(matrices['train'])
    if (scaler.var_ <= 1e-12).any():
        raise ValueError('Selected training feature has near-zero variance.')
    output = new_output(out, paths.values())
    for split, X in matrices.items():
        table = frames[split][['imageName', 'Group']].reset_index(drop=True)
        table['cohort'] = split
        table = pd.concat(
            [table, pd.DataFrame(scaler.transform(X), columns=config['features'])],
            axis=1
        )
        table.to_csv(output / f'{split}_scaled.csv', index=False)
    joblib.dump(scaler, output / 'scaler.joblib')
    write_json(output / 'preprocessing.json', dict(
        model=model_name, features=config['features'], fitted_on='train',
        inputs={s: sha256(p) for s, p in paths.items()},
        scope='Training-fitted StandardScaler applied to unscaled selected features.'
    ))
    return dict(model=model_name, features=len(config['features']), cohorts=list(paths))
