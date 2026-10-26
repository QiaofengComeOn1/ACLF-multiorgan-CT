"""Consensus K-means and random-forest cluster assignment."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def reorder_labels_by_risk(labels, event28):
    tmp = pd.DataFrame({'label': labels, 'event28': event28})
    risk_order = tmp.groupby('label')['event28'].mean().sort_values().index.tolist()
    mapping = {old: new for new, old in enumerate(risk_order)}
    new_labels = np.array([mapping[x] for x in labels])
    return (new_labels, mapping)


def preprocess_fit_transform(train_X, valid_X):
    train_X = train_X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid_X = valid_X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train = imputer.fit_transform(train_X)
    X_valid = imputer.transform(valid_X)
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    return (X_train, X_valid, imputer, scaler)


def consensus_clustering_kmeans(
    X,
    k_values=range(2, 7),
    n_repeats=1000,
    p_item=0.8,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    results = {}
    for k in k_values:
        pair_same = np.zeros((n, n), dtype=float)
        pair_together = np.zeros((n, n), dtype=float)
        for _ in range(n_repeats):
            idx = np.sort(rng.choice(n, size=max(2, int(np.ceil(n * p_item))), replace=False))
            X_sub = X[idx]
            km = KMeans(n_clusters=k, n_init=30, random_state=rng.randint(0, 10 ** 9))
            lab = km.fit_predict(X_sub)
            for i_pos, i in enumerate(idx):
                pair_together[i, idx] += 1
                same_idx = idx[lab == lab[i_pos]]
                pair_same[i, same_idx] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            consensus = np.divide(
                pair_same,
                pair_together,
                out=np.zeros_like(pair_same),
                where=pair_together > 0
            )
        np.fill_diagonal(consensus, 1.0)
        final_km = KMeans(n_clusters=k, n_init=100, random_state=random_state)
        final_labels = final_km.fit_predict(X)
        results[k] = {'consensus_matrix': consensus, 'final_labels': final_labels}
    return results


def cdf_area_from_consensus(consensus):
    vals = consensus[np.triu_indices_from(consensus, k=1)]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, None, None)
    grid = np.linspace(0, 1, 101)
    cdf = np.array([(vals <= g).mean() for g in grid])
    area = np.trapz(cdf, grid)
    return (area, grid, cdf)


def train_rf_and_predict(train_X, train_labels, valid_X, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        train_X,
        train_labels,
        test_size=0.2,
        stratify=train_labels,
        random_state=random_state
    )
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred_te = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred_te)
    cm = confusion_matrix(y_te, y_pred_te)
    valid_pred = clf.predict(valid_X)
    valid_prob = clf.predict_proba(valid_X)
    return (clf, acc, cm, valid_pred, valid_prob)
