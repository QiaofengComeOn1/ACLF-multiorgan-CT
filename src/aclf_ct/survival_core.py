"""Kaplan-Meier, log-rank and unadjusted Cox analyses."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def safe_to_numeric(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)


def format_p_value(p):
    if pd.isna(p):
        return 'NA'
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'


def format_hr_ci(hr, lo, hi):
    if pd.isna(hr) or pd.isna(lo) or pd.isna(hi):
        return 'NA'
    return f'{hr:.2f} ({lo:.2f}-{hi:.2f})'


def cluster_display_value(c, clusters=None):
    """Map zero-based cluster labels to one-based display labels."""
    if clusters is None:
        return int(c) + 1
    clusters = sorted([int(x) for x in clusters])
    if len(clusters) > 0 and min(clusters) == 0:
        return int(c) + 1
    return int(c)


def clean_base_survival_df(df, cohort_name='cohort'):
    required = ['time_days', 'event', 'cluster']
    for c in required:
        if c not in df.columns:
            raise ValueError(f'{cohort_name} 缺少必要列: {c}')
    out = df.copy()
    out['time_days'] = safe_to_numeric(out['time_days'])
    out['event'] = safe_to_numeric(out['event'])
    out['cluster'] = safe_to_numeric(out['cluster'])
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=['time_days', 'event', 'cluster'])
    out = out[out['event'].isin([0, 1])]
    out = out[out['time_days'] >= 0]
    out['time_days'] = out['time_days'].clip(lower=1e-06)
    out['event'] = out['event'].astype(int)
    out['cluster'] = out['cluster'].astype(int)
    return out


def add_endpoint_columns(df):
    out = df.copy()
    out['event_28'] = ((out['event'] == 1) & (out['time_days'] <= 28)).astype(int)
    out['time_28'] = np.minimum(out['time_days'], 28.0)
    out['time_28'] = out['time_28'].clip(lower=1e-06)
    out['event_90'] = ((out['event'] == 1) & (out['time_days'] <= 90)).astype(int)
    out['time_90'] = np.minimum(out['time_days'], 90.0)
    out['time_90'] = out['time_90'].clip(lower=1e-06)
    out['event_overall'] = out['event'].astype(int)
    out['time_overall'] = out['time_days'].clip(lower=1e-06)
    out['death90'] = out['event_90'].astype(int)
    return out


def get_endpoint_specs():
    return [
        {
            'endpoint': '28-day mortality',
            'endpoint_short': '28-day',
            'time_col': 'time_28',
            'event_col': 'event_28',
            'horizon_days': 28
        },
        {
            'endpoint': '90-day mortality',
            'endpoint_short': '90-day',
            'time_col': 'time_90',
            'event_col': 'event_90',
            'horizon_days': 90
        },
        {
            'endpoint': 'Overall mortality',
            'endpoint_short': 'Overall',
            'time_col': 'time_overall',
            'event_col': 'event_overall',
            'horizon_days': 'overall'
        }
    ]


def fit_endpoint_cox_reference(df, cohort_name='cohort', reference_cluster=None, penalizer=0.01):
    rows = []
    df = df.copy()
    clusters = sorted(df['cluster'].dropna().astype(int).unique().tolist())
    if len(clusters) < 2:
        raise ValueError(f'{cohort_name} 中 cluster 数量不足，无法做 Cox 分析。')
    if reference_cluster is None:
        reference_cluster = clusters[0]
    else:
        reference_cluster = int(reference_cluster)
        if reference_cluster not in clusters:
            raise ValueError(f'{cohort_name} 中不存在 reference_cluster={reference_cluster}')
    ref_disp = cluster_display_value(reference_cluster, clusters)
    for spec in get_endpoint_specs():
        endpoint = spec['endpoint']
        endpoint_short = spec['endpoint_short']
        time_col = spec['time_col']
        event_col = spec['event_col']
        horizon_days = spec['horizon_days']
        model_df = df[[time_col, event_col, 'cluster']].copy()
        model_df = model_df.rename(columns={time_col: 'time', event_col: 'event'})
        model_df = model_df.replace([np.inf, -np.inf], np.nan)
        model_df = model_df.dropna(subset=['time', 'event', 'cluster'])
        model_df['time'] = safe_to_numeric(model_df['time']).clip(lower=1e-06)
        model_df['event'] = safe_to_numeric(model_df['event']).astype(int)
        model_df['cluster'] = safe_to_numeric(model_df['cluster']).astype(int)
        cluster_n = model_df.groupby('cluster').size().to_dict()
        cluster_events = model_df.groupby('cluster')['event'].sum().to_dict()
        design = model_df[['time', 'event', 'cluster']].copy()
        term_meta = []
        for c in clusters:
            if c == reference_cluster:
                continue
            c_disp = cluster_display_value(c, clusters)
            term = f'Cluster_{c_disp}_vs_{ref_disp}'
            design[term] = (design['cluster'] == c).astype(int)
            term_meta.append({
                'term': term,
                'comparison_cluster_raw': c,
                'comparison_cluster_display': c_disp
            })
        covariates = [m['term'] for m in term_meta]
        cox_df = design[['time', 'event'] + covariates].copy()
        cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna()
        usable_covariates = []
        for c in covariates:
            if cox_df[c].nunique(dropna=True) >= 2:
                usable_covariates.append(c)
        cox_df = cox_df[['time', 'event'] + usable_covariates].copy()
        model_status = 'not_fitted'
        cph = None
        if len(cox_df) >= 10 and int(cox_df['event'].sum()) >= 2 and (len(usable_covariates) > 0):
            try:
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col='time', event_col='event')
                model_status = 'fitted'
            except Exception:
                try:
                    cph = CoxPHFitter(penalizer=penalizer)
                    cph.fit(cox_df, duration_col='time', event_col='event')
                    model_status = f'fitted_with_penalizer_{penalizer}'
                except Exception:
                    cph = None
                    model_status = 'fit_failed'
        ref_n = int(cluster_n.get(reference_cluster, 0))
        ref_events = int(cluster_events.get(reference_cluster, 0))
        for m in term_meta:
            term = m['term']
            c = m['comparison_cluster_raw']
            c_disp = m['comparison_cluster_display']
            cmp_n = int(cluster_n.get(c, 0))
            cmp_events = int(cluster_events.get(c, 0))
            if term not in usable_covariates:
                estimable = False
                reason = 'constant_or_all_missing'
            elif cph is None:
                estimable = False
                reason = model_status
            elif term not in cph.summary.index:
                estimable = False
                reason = 'not_in_model_summary'
            else:
                estimable = True
                reason = 'estimated'
            if estimable:
                s = cph.summary.loc[term]
                hr = float(s['exp(coef)'])
                ci_low = float(s['exp(coef) lower 95%'])
                ci_high = float(s['exp(coef) upper 95%'])
                p_value = float(s['p'])
            else:
                hr = np.nan
                ci_low = np.nan
                ci_high = np.nan
                p_value = np.nan
            rows.append({
                'cohort': cohort_name,
                'endpoint': endpoint,
                'endpoint_short': endpoint_short,
                'horizon_days': horizon_days,
                'reference_cluster_raw': reference_cluster,
                'reference_cluster_display': ref_disp,
                'comparison_cluster_raw': c,
                'comparison_cluster_display': c_disp,
                'endpoint_label': f'{endpoint_short} (Cluster {ref_disp}: Ref)',
                'comparison_label': f'Cluster {c_disp} vs Ref',
                'n_reference': ref_n,
                'events_reference': ref_events,
                'event_rate_reference_percent': ref_events / ref_n * 100.0 if ref_n > 0 else np.nan,
                'n_comparison': cmp_n,
                'events_comparison': cmp_events,
                'event_rate_comparison_percent': cmp_events / cmp_n * 100.0 if cmp_n > 0 else np.nan,
                'events_n_reference': f'{ref_events}/{ref_n}',
                'events_n_comparison': f'{cmp_events}/{cmp_n}',
                'events_n_text': f'{cmp_events}/{cmp_n} vs {ref_events}/{ref_n}',
                'hr': hr,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_value': p_value,
                'HR (95% CI)': format_hr_ci(hr, ci_low, ci_high),
                'P-value': format_p_value(p_value),
                'estimable': estimable,
                'reason': reason,
                'model_status': model_status,
                'n_model': int(len(cox_df)),
                'n_events_model': int(cox_df['event'].sum()) if len(cox_df) > 0 else 0
            })
    return pd.DataFrame(rows)
