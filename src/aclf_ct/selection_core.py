"""Univariate filtering, correlation filtering and L1 feature selection."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline


def fit_clean_transform_train(X_train_num: pd.DataFrame):
    all_nan_cols = [c for c in X_train_num.columns if X_train_num[c].isna().all()]
    X0 = X_train_num.drop(columns=all_nan_cols)
    if X0.shape[1] == 0:
        raise ValueError('训练集去除全空列后无剩余特征。')
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X0), columns=X0.columns, index=X0.index)
    nunique = X_imp.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X_final = X_imp.drop(columns=constant_cols)
    if X_final.shape[1] == 0:
        raise ValueError('训练集清洗后特征为空（可能全为空或常数列）。')
    cleaner = {
        'imputer': imputer,
        'kept_cols_after_clean': list(X_final.columns),
        'dropped_all_nan': all_nan_cols,
        'dropped_constant': constant_cols
    }
    return (X_final, cleaner)


def transform_other_set(X_num: pd.DataFrame, cleaner: dict):
    imputer = cleaner['imputer']
    fit_cols = imputer.feature_names_in_.tolist()
    X_align = X_num.copy()
    for c in fit_cols:
        if c not in X_align.columns:
            X_align[c] = np.nan
    X_align = X_align[fit_cols]
    X_imp = pd.DataFrame(imputer.transform(X_align), columns=fit_cols, index=X_align.index)
    kept_cols = cleaner['kept_cols_after_clean']
    X_final = X_imp[kept_cols].copy()
    return X_final


def utest_fdr_filter(X_train: pd.DataFrame, y_train: np.ndarray, alpha=0.05):
    results = []
    for col in X_train.columns:
        x0 = X_train.loc[y_train == 0, col].dropna()
        x1 = X_train.loc[y_train == 1, col].dropna()
        if len(x0) == 0 or len(x1) == 0:
            continue
        try:
            u_stat, p_val = mannwhitneyu(x0, x1, alternative='two-sided')
        except Exception:
            u_stat, p_val = (np.nan, 1.0)
        if not np.isfinite(p_val):
            p_val = 1.0
        results.append([col, format_median_iqr(x0), format_median_iqr(x1), u_stat, p_val])
    if len(results) == 0:
        result_df = pd.DataFrame(columns=[
            'Variable',
            'Group0_Median(Q1,Q3)',
            'Group1_Median(Q1,Q3)',
            'U',
            'P_raw',
            'P_FDR',
            'Significant_FDR'
        ])
        return ([], result_df)
    result_df = pd.DataFrame(
        results,
        columns=['Variable', 'Group0_Median(Q1,Q3)', 'Group1_Median(Q1,Q3)', 'U', 'P_raw']
    )
    reject, p_fdr, _, _ = multipletests(result_df['P_raw'].values, alpha=alpha, method='fdr_bh')
    result_df['P_FDR'] = p_fdr
    result_df['Significant_FDR'] = reject
    result_df = result_df.sort_values(by='P_raw', ascending=True).reset_index(drop=True)
    kept = result_df.loc[result_df['P_FDR'] < alpha, 'Variable'].tolist()
    return (kept, result_df)


def spearman_filter(X_train: pd.DataFrame, corr_threshold=0.9):
    if X_train.shape[1] <= 1:
        return (list(X_train.columns), pd.DataFrame(), [])
    corr_df = X_train.corr(method='spearman')
    upper = corr_df.abs().where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    selected_cols = [col for col in X_train.columns if col not in to_drop]
    if len(selected_cols) == 0:
        selected_cols = [X_train.columns[0]]
    return (selected_cols, corr_df, to_drop)


def logistic_l1_cv_select(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    Cs=None,
    cv_folds=10,
    random_state=42,
    max_iter=20000,
    n_jobs=8,
    class_weight='balanced',
    use_C_rule='1se'
):
    if Cs is None:
        Cs = np.logspace(-4, 2, 40)
    Cs = np.asarray(Cs, dtype=float)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            penalty='l1', solver='saga', max_iter=max_iter,
            random_state=random_state, class_weight=class_weight
        ))
    ])
    logit_cv = GridSearchCV(
        pipeline,
        param_grid={'model__C': Cs.tolist()},
        cv=cv,
        scoring='roc_auc',
        n_jobs=n_jobs,
        refit=False,
        error_score='raise'
    )
    logit_cv.fit(X_train, y_train)
    scores = np.vstack([logit_cv.cv_results_[f'split{i}_test_score']
                        for i in range(cv_folds)])
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite cross-validated selection scores.')
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)
    idx_best = int(np.argmax(mean_scores))
    C_best = float(Cs[idx_best])
    threshold_1se = mean_scores[idx_best] - std_scores[idx_best]
    candidates = np.where(mean_scores >= threshold_1se)[0]
    idx_1se = int(candidates[np.argmin(Cs[candidates])])
    C_1se = float(Cs[idx_1se])
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_train),
                             columns=X_train.columns, index=X_train.index)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    def fit_one_C(C_value):
        model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=float(C_value),
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight
        )
        model.fit(X_scaled, y_train)
        coef = pd.Series(model.coef_.ravel(), index=X_train.columns)
        selected = coef[coef != 0].index.tolist()
        return (model, coef, selected)
    model_best, coef_best, sel_best = fit_one_C(C_best)
    model_1se, coef_1se, sel_1se = fit_one_C(C_1se)
    if use_C_rule == 'best':
        final_model = model_best
        final_coef = coef_best
        final_selected = sel_best
        C_used = C_best
    else:
        final_model = model_1se
        final_coef = coef_1se
        final_selected = sel_1se
        C_used = C_1se
    return {
        'imputer': imputer,
        'scaler': scaler,
        'logit_cv': logit_cv,
        'model_final': final_model,
        'coef_series': final_coef,
        'selected_features': final_selected,
        'intercept': float(final_model.intercept_[0]),
        'C_best': C_best,
        'C_1se': C_1se,
        'C_used': C_used,
        'mean_scores': mean_scores,
        'std_scores': std_scores
    }


def select_features_one_fold(X_fold_tr, y_fold_tr, params, X_fold_val=None):
    X_clean, cleaner = fit_clean_transform_train(X_fold_tr)
    kept_u, _ = utest_fdr_filter(X_clean, y_fold_tr, alpha=params['fdr_alpha'])
    if not kept_u:
        return []
    X1 = X_clean[kept_u].copy()
    kept_s, _, _ = spearman_filter(X1, corr_threshold=params['corr_threshold'])
    X2 = X_fold_tr[kept_s].copy()
    logit_res = logistic_l1_cv_select(
        X_train=X2,
        y_train=y_fold_tr,
        Cs=params['Cs'],
        cv_folds=params['logit_cv_folds'],
        random_state=params['random_state'],
        max_iter=params['logit_max_iter'],
        n_jobs=params['logit_n_jobs'],
        class_weight=params['logit_class_weight'],
        use_C_rule=params['use_C_rule']
    )
    feats = logit_res['selected_features']
    if X_fold_val is not None and feats:
        cleaned = transform_other_set(X_fold_val, cleaner)
        transformed = logit_res['scaler'].transform(cleaned[kept_s])
        selected = transformed[:, [kept_s.index(name) for name in feats]]
        if not np.isfinite(selected).all():
            raise ValueError('Nonfinite selected features in a held-out partition.')
    return feats


def format_median_iqr(series):
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f'{median:.4f} ({q1:.4f}, {q3:.4f})'
