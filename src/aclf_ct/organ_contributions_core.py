"""Organ-level sensitivity analysis of cluster assignments."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, adjusted_rand_score,
    normalized_mutual_info_score, accuracy_score, confusion_matrix)
CV_SPLITS = 5


def identify_organ_features(columns):
    liver_cols = [c for c in columns if c.startswith('Liver_f_')]
    spleen_cols = [c for c in columns if c.startswith('Spleen_f_')]
    muscle_cols = [c for c in columns if c.startswith('Muscle_f_')]
    return (liver_cols, spleen_cols, muscle_cols)


def prepare_X(df, cols):
    X = df[cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled


def run_pca(X, n_components=50, random_state=42):
    if 0 < n_components < 1:
        pca = PCA(n_components=n_components, svd_solver='full', random_state=random_state)
    else:
        n_components = min(n_components, X.shape[0], X.shape[1])
        if n_components < 1:
            raise ValueError('No valid PCA component available.')
        pca = PCA(n_components=n_components, random_state=random_state)
    Xp = pca.fit_transform(X)
    return (Xp, pca)


def run_kmeans(X, k, random_state=42, n_init=50):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    return (labels, km)


def contingency_df(y_true, y_pred):
    return pd.crosstab(
        pd.Series(y_true, name='Fusion subtype'),
        pd.Series(y_pred, name='Organ-specific subtype')
    )


def optimal_label_mapping(y_ref, y_pred):
    """Align cluster labels by Hungarian matching for graphical display."""
    ct = pd.crosstab(pd.Series(y_ref), pd.Series(y_pred))
    rows = sorted(ct.index.tolist())
    cols = sorted(ct.columns.tolist())
    mat = ct.loc[rows, cols].values
    max_val = mat.max() if mat.size > 0 else 0
    cost = max_val - mat
    r_ind, c_ind = linear_sum_assignment(cost)
    mapping = {}
    for r, c in zip(r_ind, c_ind):
        mapping[cols[c]] = rows[r]
    unmatched = [c for c in cols if c not in mapping]
    used = set(mapping.values())
    next_lab = max(max(rows) if len(rows) > 0 else -1, max(cols) if len(cols) > 0 else -1) + 1
    for c in unmatched:
        while next_lab in used:
            next_lab += 1
        mapping[c] = next_lab
        used.add(next_lab)
    y_new = np.array([mapping[v] for v in y_pred])
    return (y_new, mapping)


def evaluate_clustering_against_fusion(X, y_ref, k, n_components=50):
    Xp, _ = run_pca(X, n_components=n_components, random_state=RANDOM_STATE)
    y_pred, _ = run_kmeans(Xp, k=k, random_state=RANDOM_STATE, n_init=N_INIT)
    ari = adjusted_rand_score(y_ref, y_pred)
    nmi = normalized_mutual_info_score(y_ref, y_pred)
    return {'Xp': Xp, 'labels_raw': y_pred, 'ARI': ari, 'NMI': nmi}


def cv_bal_acc(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        scores.append(balanced_accuracy_score(y[te], pred))
    return float(np.mean(scores))


def block_permutation_importance(X_blocks, y, n_perm=200, seed=2026):
    rng = np.random.RandomState(seed)
    order = ['liver', 'spleen', 'muscle']
    X_all = np.hstack([X_blocks[k] for k in order])
    baseline = cv_bal_acc(X_all, y, n_splits=CV_SPLITS, seed=seed)
    nL = X_blocks['liver'].shape[1]
    nS = X_blocks['spleen'].shape[1]
    nM = X_blocks['muscle'].shape[1]
    block_ranges = {'liver': (0, nL), 'spleen': (nL, nL + nS), 'muscle': (nL + nS, nL + nS + nM)}
    idx = np.arange(len(y))
    rows = []
    for blk in ['liver', 'spleen', 'muscle']:
        s, e = block_ranges[blk]
        drops = []
        for i in range(n_perm):
            Xp = X_all.copy()
            perm = rng.permutation(idx)
            Xp[:, s:e] = Xp[perm, s:e]
            acc_perm = cv_bal_acc(Xp, y, n_splits=CV_SPLITS, seed=seed + i + 1)
            drops.append(baseline - acc_perm)
        drops = np.array(drops, dtype=float)
        rows.append({
            'Organ': blk.capitalize(),
            'Baseline_Balanced_Accuracy': baseline,
            'Importance': np.mean(drops),
            'CI_low': np.quantile(drops, 0.025),
            'CI_high': np.quantile(drops, 0.975)
        })
    return pd.DataFrame(rows)

RANDOM_STATE = 42
PCA_N = 50
N_INIT = 50
