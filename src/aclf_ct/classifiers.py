"""Classifier definitions and hyperparameters."""
import os
import re
import json
import joblib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
RANDOM_STATE = 0


def pick_safe_n_jobs() -> int:
    return 1


class PygamLogisticGAMClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn-compatible wrapper for pygam.LogisticGAM."""

    def __init__(self, max_iter=200, lam=0.6):
        self.max_iter = max_iter
        self.lam = lam

    def fit(self, X, y):
        from pygam import LogisticGAM
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.model_ = LogisticGAM(lam=self.lam, max_iter=self.max_iter)
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        X = np.asarray(X, dtype=float)
        p1 = np.asarray(self.model_.predict_proba(X)).reshape(-1)
        p1 = np.clip(p1, 0, 1)
        return np.c_[1 - p1, p1]

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    """Rotation Forest with groupwise PCA and averaged tree probabilities."""

    def __init__(
        self,
        n_estimators=50,
        min_group=2,
        max_group=4,
        bootstrap=True,
        max_depth=None,
        random_state=0
    ):
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_groups(self, n_features, rng):
        feat_idx = np.arange(n_features)
        rng.shuffle(feat_idx)
        groups = []
        i = 0
        while i < n_features:
            gsize = rng.randint(self.min_group, self.max_group + 1)
            group = feat_idx[i:i + gsize]
            if len(group) == 0:
                break
            groups.append(group)
            i += gsize
        return groups

    def _fit_rotation_matrix(self, X, rng):
        n_samples, n_features = X.shape
        groups = self._make_groups(n_features, rng)
        R = np.eye(n_features)
        for group in groups:
            cols = np.array(group, dtype=int)
            Xg = X[:, cols]
            if Xg.shape[1] <= 1:
                R[np.ix_(cols, cols)] = np.eye(len(cols))
                continue
            if np.all(np.std(Xg, axis=0) < 1e-12):
                R[np.ix_(cols, cols)] = np.eye(len(cols))
                continue
            try:
                pca = PCA(n_components=len(cols), random_state=rng.randint(0, 10 ** 9))
                pca.fit(Xg)
                comp = pca.components_.T
                if comp.shape != (len(cols), len(cols)):
                    comp = np.eye(len(cols))
            except Exception:
                comp = np.eye(len(cols))
            R[np.ix_(cols, cols)] = comp
        return R

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.estimators_ = []
        self.rotation_matrices_ = []
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = n_features
        for i in range(self.n_estimators):
            tree_rng = np.random.RandomState(rng.randint(0, 10 ** 9))
            if self.bootstrap:
                idx = tree_rng.randint(0, n_samples, n_samples)
            else:
                idx = np.arange(n_samples)
            X_boot = X[idx]
            y_boot = y[idx]
            R = self._fit_rotation_matrix(X_boot, tree_rng)
            X_rot = X_boot @ R
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=tree_rng.randint(0, 10 ** 9)
            )
            tree.fit(X_rot, y_boot)
            self.rotation_matrices_.append(R)
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['estimators_', 'rotation_matrices_'])
        X = np.asarray(X, dtype=float)
        probs = []
        for tree, R in zip(self.estimators_, self.rotation_matrices_):
            X_rot = X @ R
            p = tree.predict_proba(X_rot)
            if p.shape[1] == 1:
                cls = tree.classes_[0]
                if cls == 0:
                    p = np.c_[p[:, 0], np.zeros(len(X))]
                else:
                    p = np.c_[np.zeros(len(X)), p[:, 0]]
            elif list(tree.classes_) != [0, 1]:
                p2 = np.zeros((len(X), 2), dtype=float)
                for j, c in enumerate(tree.classes_):
                    p2[:, int(c)] = p[:, j]
                p = p2
            probs.append(p)
        mean_p = np.mean(probs, axis=0)
        mean_p = np.clip(mean_p, 0, 1)
        row_sum = mean_p.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        mean_p = mean_p / row_sum
        return mean_p

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)


def build_models(requested=None) -> Dict[str, object]:
    """Construct the requested classifiers and their hyperparameters."""
    models = {}
    models['LR'] = LogisticRegression(max_iter=5000, solver='liblinear', random_state=RANDOM_STATE)
    models['SVM'] = SVC(kernel='linear', probability=False, random_state=RANDOM_STATE)
    models['SGD'] = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        max_iter=3000,
        tol=0.001,
        random_state=RANDOM_STATE
    )
    models['RF'] = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=pick_safe_n_jobs(),
        class_weight='balanced'
    )
    models['KNN'] = KNeighborsClassifier(n_neighbors=5, weights='distance')
    models['GBDT'] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    models['DT'] = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=pick_safe_n_jobs(),
        class_weight='balanced'
    )
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE)
    models['NB'] = GaussianNB()
    models['MLP'] = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    models['GPC'] = GaussianProcessClassifier(
        kernel=C(1.0) * RBF(length_scale=1.0),
        random_state=RANDOM_STATE,
        n_restarts_optimizer=0,
        max_iter_predict=100,
        n_jobs=pick_safe_n_jobs()
    )
    models['RotF'] = RotationForestClassifier(
        n_estimators=100,
        min_group=2,
        max_group=4,
        bootstrap=True,
        max_depth=None,
        random_state=RANDOM_STATE
    )
    if requested is None or 'GAM' in requested:
        import pygam
        models['GAM'] = PygamLogisticGAMClassifier(max_iter=200, lam=0.6)
    if requested is None or 'XGB' in requested:
        from xgboost import XGBClassifier
        models['XGB'] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=pick_safe_n_jobs(),
            use_label_encoder=False
        )
    if requested is None or 'LGB' in requested:
        import lightgbm as lgb
        models['LGB'] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=pick_safe_n_jobs()
        )
    return models if requested is None else {name: models[name] for name in requested}


def get_proba_1(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1]
        return np.clip(p, 0, 1)
    if hasattr(model, 'decision_function'):
        z = model.decision_function(X)
        p = 1 / (1 + np.exp(-z))
        return np.clip(p, 0, 1)
    yhat = model.predict(X).astype(float)
    return np.clip(yhat, 0, 1)
