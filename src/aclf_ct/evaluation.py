"""Performance evaluation of saved models and predictions."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .io import (binary_labels, probabilities, read_table, new_output, profile,
                 selected_matrix, check_disjoint, sha256, write_json)
from .metrics import (calc_metrics, compute_auc_ci, find_best_threshold_by_youden,
                      calibration_curve_stats, delong_pvalue, dca_net_benefit)


def brier_interval(y, p, repetitions=2000, seed=20260915):
    """Patient-level percentile bootstrap interval for the Brier score."""
    squared = (np.asarray(p) - np.asarray(y)) ** 2
    rng = np.random.default_rng(seed)
    bootstrap = squared[rng.integers(0, len(y), size=(repetitions, len(y)))].mean(axis=1)
    low, high = np.quantile(bootstrap, [.025, .975])
    return float(squared.mean()), float(low), float(high)


def evaluate_vector(y, p, threshold, bootstraps=2000, auc_seed=42):
    y = binary_labels(y)
    p = probabilities(p, len(y))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be a fixed probability in [0, 1].')
    result = calc_metrics(y, (p >= threshold).astype(int), p)
    _, lower, upper = compute_auc_ci(y, p, n_bootstraps=bootstraps, seed=auc_seed)
    brier, brier_low, brier_high = brier_interval(y, p, repetitions=bootstraps)
    result.update(n=len(y), events=int(y.sum()), threshold=float(threshold), auc_bootstrap_seed=int(auc_seed),
                  AUC_low=lower, AUC_high=upper, Brier=brier,
                  Brier_low=brier_low, Brier_high=brier_high)
    return result


def evaluate_runs(runs_root, out, trusted=False, bootstraps=2000):
    """Evaluate fitted models in the six configured run directories."""
    if not trusted:
        raise ValueError('Only author-controlled model files may be loaded; pass --trust-model-files.')
    import joblib
    from .classifiers import get_proba_1
    runs_root = Path(runs_root)
    configs = profile()
    rows, curves, net_benefits, prediction_frames, manifest = [], [], [], {}, []
    for name, config in configs.items():
        run = runs_root / config['source_run']
        frames = {s: read_table(run / 'preprocess' / f'{s}_scaled.csv')
                  for s in ('train', 'internal', 'external')}
        check_disjoint(frames.values())
        for split, frame in frames.items():
            X, y = selected_matrix(frame, config['features'])
            if len(y) != config['counts'][split] or int(y.sum()) != config['events'][split]:
                raise ValueError(f'{name}/{split}: cohort counts differ from the configured profile.')
            manifest.append({'input': f'{name}/{split}', 'sha256': sha256(run / 'preprocess' / f'{split}_scaled.csv'),
                             'n':len(y), 'events':int(y.sum()), 'features':X.shape[1]})
        leaderboard_path = run / 'train' / 'leaderboard.json'
        leaderboard = json.loads(leaderboard_path.read_text(encoding='utf-8'))
        if leaderboard['best_raw_model'] != config['classifier']:
            raise ValueError(f'{name}: saved classifier differs from the configured profile.')
        calibrated = bool(leaderboard['calibration']['best_calibrated_exists'])
        if calibrated != config['use_calibrated_probabilities']:
            raise ValueError(f'{name}: saved calibration status differs from the configured profile.')
        filename = 'best_calibrated_model.pkl' if calibrated else f"best_raw_model_{config['classifier']}.pkl"
        model_path = run / 'train' / filename
        model = joblib.load(model_path)
        manifest.append({
            'input': f'{name}/model',
            'sha256': sha256(model_path),
            'calibrated':calibrated
        })
        X, y = selected_matrix(frames['train'], config['features'])
        threshold, _ = find_best_threshold_by_youden(y, get_proba_1(model, X))
        for split in ('internal', 'external'):
            X, y = selected_matrix(frames[split], config['features'])
            p = probabilities(get_proba_1(model, X), len(y))
            rows.append(dict(model=name, cohort=split, **evaluate_vector(y, p, threshold, bootstraps,
                             auc_seed=config['auc_bootstrap_seeds'][split])))
            predicted, observed, lower, upper = calibration_curve_stats(y, p)
            curves.extend(dict(model=name, cohort=split, bin=i, predicted=a, observed=b, lower=c, upper=d)
                          for i, (a,b,c,d) in enumerate(zip(predicted,observed,lower,upper)))
            thresholds = np.linspace(.01, .99, 99)
            benefits = dca_net_benefit(y, p, thresholds)
            net_benefits.extend(dict(model=name, cohort=split, threshold=t, net_benefit=b)
                                for t,b in zip(thresholds,benefits))
            prediction_frames[name, split] = pd.DataFrame({'id': frames[split]['imageName'].astype(str), 'y':y, 'p':p})
    comparisons = []
    for split in ('internal', 'external'):
        for first, second in combinations(configs, 2):
            a, b = prediction_frames[first, split], prediction_frames[second, split]
            joined = a.merge(b, on='id', validate='one_to_one', suffixes=('_a','_b'))
            if len(joined) != len(a) or len(a) != len(b) or not np.array_equal(joined.y_a, joined.y_b):
                raise ValueError('Paired model comparison requires identical cases and outcomes.')
            pvalue = delong_pvalue(
                joined.y_a.to_numpy(),
                joined.p_a.to_numpy(),
                joined.p_b.to_numpy()
            )
            comparisons.append(dict(cohort=split, first=first, second=second, p_value=pvalue))
    output = new_output(out, [runs_root])
    pd.DataFrame(rows).to_csv(output/'performance.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output/'delong.csv', index=False)
    pd.DataFrame(curves).to_csv(output/'calibration.csv', index=False)
    pd.DataFrame(net_benefits).to_csv(output/'decision_curves.csv', index=False)
    write_json(output/'input_manifest.json', manifest)
    return pd.DataFrame(rows)


def evaluate_fixed_file(path, out, threshold=.5, bootstraps=2000):
    frame = read_table(path)
    result = evaluate_vector(frame['y_true'], frame['p_aclf'], threshold, bootstraps)
    output = new_output(out, [path])
    write_json(output/'performance.json', result)
    write_json(output/'input_manifest.json', {'sha256':sha256(path)})
    return result
