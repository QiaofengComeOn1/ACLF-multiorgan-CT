"""Discrimination, calibration and decision-curve metrics."""

from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    out = {}
    out['AUC'] = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) == 2 else float('nan')
    out['Accuracy'] = float(accuracy_score(y_true, y_pred))
    out['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    out['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    out['F1'] = float(f1_score(y_true, y_pred, zero_division=0))
    out['Brier'] = float(brier_score_loss(y_true, y_proba))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out['TN'] = float(tn)
    out['FP'] = float(fp)
    out['FN'] = float(fn)
    out['TP'] = float(tp)
    out['Sensitivity'] = float(tp / (tp + fn + 1e-12))
    out['Specificity'] = float(tn / (tn + fp + 1e-12))
    return out


def compute_auc_ci(y_true, y_score, n_bootstraps=2000, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return (float('nan'), None, None)
    auc_value = roc_auc_score(y_true, y_score)
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(int(n_bootstraps)):
        idx = rng.randint(0, len(y_true), len(y_true))
        y_bs = y_true[idx]
        s_bs = y_score[idx]
        if len(np.unique(y_bs)) < 2:
            continue
        aucs.append(roc_auc_score(y_bs, s_bs))
    if len(aucs) == 0:
        return (float(auc_value), None, None)
    lower = float(np.percentile(aucs, 2.5))
    upper = float(np.percentile(aucs, 97.5))
    return (float(auc_value), lower, upper)


def find_best_threshold_by_youden(y_true, y_proba):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return (
            0.5,
            {
                'best_threshold': 0.5,
                'best_youden': None,
                'sensitivity': None,
                'specificity': None,
                'n_candidates': 0
            }
        )
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, drop_intermediate=False)
    youden = tpr - fpr
    finite_mask = np.isfinite(thresholds)
    if finite_mask.sum() == 0:
        return (
            0.5,
            {
                'best_threshold': 0.5,
                'best_youden': None,
                'sensitivity': None,
                'specificity': None,
                'n_candidates': 0
            }
        )
    fpr_f = fpr[finite_mask]
    tpr_f = tpr[finite_mask]
    thresholds_f = thresholds[finite_mask]
    youden_f = youden[finite_mask]
    spec_f = 1.0 - fpr_f
    best_youden = np.max(youden_f)
    candidate_idx = np.where(np.isclose(youden_f, best_youden, atol=1e-12))[0]
    if len(candidate_idx) == 1:
        best_idx = int(candidate_idx[0])
    else:
        balance = np.abs(tpr_f[candidate_idx] - spec_f[candidate_idx])
        best_idx = int(candidate_idx[np.argmin(balance)])
    best_thr = float(thresholds_f[best_idx])
    best_sens = float(tpr_f[best_idx])
    best_spec = float(spec_f[best_idx])
    return (
        best_thr,
        {
            'best_threshold': best_thr,
            'best_youden': float(best_youden),
            'sensitivity': best_sens,
            'specificity': best_spec,
            'n_candidates': int(len(thresholds_f))
        }
    )


def calibration_curve_stats(y_true, y_proba, n_bins=10, with_ci=True):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    frac_pos, mean_pred, ci_low, ci_high = ([], [], [], [])
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            frac_pos.append(np.nan)
            mean_pred.append(bin_centers[b])
            if with_ci:
                ci_low.append(np.nan)
                ci_high.append(np.nan)
            continue
        y_bin = y_true[mask]
        p_hat = y_bin.mean()
        frac_pos.append(p_hat)
        mean_pred.append(y_proba[mask].mean())
        if with_ci:
            n = len(y_bin)
            se = np.sqrt(p_hat * (1 - p_hat) / max(n, 1))
            ci_low.append(max(0, p_hat - 1.96 * se))
            ci_high.append(min(1, p_hat + 1.96 * se))
    mean_pred = np.array(mean_pred, dtype=float)
    frac_pos = np.array(frac_pos, dtype=float)
    if with_ci:
        return (mean_pred, frac_pos, np.array(ci_low, dtype=float), np.array(ci_high, dtype=float))
    else:
        return (mean_pred, frac_pos, None, None)


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank for ties."""
    x = np.asarray(x)
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1.0
        T[i:j] = mid
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _fast_delong(preds_sorted_transposed: np.ndarray, label_1_count: int):
    """
    Fast DeLong for AUC variance/cov.
    preds_sorted_transposed: shape (k_classifiers, n_examples),
    sorted by ground-truth with positives first.
    """
    m = int(label_1_count)
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]
    tx = np.zeros((k, m), dtype=float)
    ty = np.zeros((k, n), dtype=float)
    tz = np.zeros((k, m + n), dtype=float)
    for r in range(k):
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
        tx[r, :] = _compute_midrank(preds_sorted_transposed[r, :m])
        ty[r, :] = _compute_midrank(preds_sorted_transposed[r, m:])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return (aucs, delong_cov)


def delong_pvalue(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> float:
    """Return two-sided p-value for AUC difference using DeLong test."""
    y_true = np.asarray(y_true).astype(int)
    pred1 = np.asarray(pred1).astype(float)
    pred2 = np.asarray(pred2).astype(float)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    p1_sorted = pred1[order]
    p2_sorted = pred2[order]
    m = int(y_sorted.sum())
    if m == 0 or m == len(y_sorted):
        return float('nan')
    preds = np.vstack([p1_sorted, p2_sorted])
    aucs, cov = _fast_delong(preds, label_1_count=m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float('nan')
    z = diff / np.sqrt(var)
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return float(p)


def dca_net_benefit(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    NB(pt) = TP/N - FP/N * (pt/(1-pt))
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    N = len(y_true)
    nb = np.zeros_like(thresholds, dtype=float)
    for k, pt in enumerate(thresholds):
        pred = (y_proba >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        nb[k] = tp / N - fp / N * (pt / (1 - pt + 1e-12))
    return nb
