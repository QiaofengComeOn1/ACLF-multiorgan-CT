"""Global and exploratory pairwise clinical comparisons across fixed imaging clusters."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kruskal,chi2_contingency,mannwhitneyu
from .survival_core import cluster_display_value
from .io import read_table,check_ids,new_output,write_json,sha256


def p_to_stars(p):
    if pd.isna(p):
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'

def get_default_clinical_vars():
    return [
        'sex',
        'age',
        'WBC',
        'HGB',
        'PLT',
        'ALT',
        'AST',
        'TBIL',
        'DBIL',
        'ALB',
        'SCR',
        'K',
        'Na',
        'PT',
        'INR',
        'Liver_Volume_ml',
        'Spleen_Volume_ml',
        'Muscle_Volume_ml',
        'Max_Diameter_mm'
    ]

def compute_clinical_comparison_table(df, clinical_vars):
    rows = []
    if 'cluster' not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp['cluster'] = pd.to_numeric(tmp['cluster'], errors='coerce')
    tmp = tmp.dropna(subset=['cluster'])
    tmp['cluster'] = tmp['cluster'].astype(int)
    clusters = sorted(tmp['cluster'].unique())
    for var in clinical_vars:
        if var not in tmp.columns:
            continue
        row = {'variable': var}
        if var == 'sex':
            dat = tmp[['cluster', var]].copy()
            dat[var] = pd.to_numeric(dat[var], errors='coerce')
            dat = dat.dropna()
            contingency = pd.crosstab(dat['cluster'], dat[var])
            p_value = np.nan
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                try:
                    _, p_value, _, _ = chi2_contingency(contingency)
                except Exception:
                    p_value = np.nan
            row['test'] = 'chi_square'
            row['p_value'] = p_value
            for c in clusters:
                sub = dat[dat['cluster'] == c][var]
                n = len(sub)
                pos = int((sub == 1).sum()) if n > 0 else np.nan
                rate = pos / n * 100 if n > 0 else np.nan
                row[f'Cluster_{cluster_display_value(c, clusters)}'] = f'{pos}/{n} ({rate:.1f}%)' if n > 0 else 'NA'
        else:
            dat = tmp[['cluster', var]].copy()
            dat[var] = pd.to_numeric(dat[var], errors='coerce')
            dat = dat.dropna()
            groups = [dat.loc[dat['cluster'] == c, var].values for c in clusters]
            p_value = np.nan
            if sum([len(g) > 0 for g in groups]) >= 2:
                try:
                    _, p_value = kruskal(*groups)
                except Exception:
                    p_value = np.nan
            row['test'] = 'kruskal'
            row['p_value'] = p_value
            for c in clusters:
                sub = dat.loc[dat['cluster'] == c, var]
                if len(sub) > 0:
                    med = float(np.median(sub))
                    q1 = float(np.percentile(sub, 25))
                    q3 = float(np.percentile(sub, 75))
                    row[f'Cluster_{cluster_display_value(c, clusters)}'] = f'{med:.3f} ({q1:.3f}-{q3:.3f})'
                else:
                    row[f'Cluster_{cluster_display_value(c, clusters)}'] = 'NA'
        rows.append(row)
    return pd.DataFrame(rows)

def compute_pairwise_mwu(df, clinical_vars):
    rows = []
    if 'cluster' not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp['cluster'] = pd.to_numeric(tmp['cluster'], errors='coerce')
    tmp = tmp.dropna(subset=['cluster'])
    tmp['cluster'] = tmp['cluster'].astype(int)
    clusters = sorted(tmp['cluster'].unique())
    for var in clinical_vars:
        if var == 'sex':
            continue
        if var not in tmp.columns:
            continue
        dat = tmp[['cluster', var]].copy()
        dat[var] = pd.to_numeric(dat[var], errors='coerce')
        dat = dat.dropna()
        for c1, c2 in combinations(clusters, 2):
            x = dat.loc[dat['cluster'] == c1, var].values
            y = dat.loc[dat['cluster'] == c2, var].values
            p_value = np.nan
            stat = np.nan
            if len(x) > 0 and len(y) > 0:
                try:
                    stat, p_value = mannwhitneyu(x, y, alternative='two-sided')
                except Exception:
                    pass
            rows.append({
                'variable': var,
                'cluster_a': cluster_display_value(c1, clusters),
                'cluster_b': cluster_display_value(c2, clusters),
                'test': 'mannwhitneyu',
                'statistic': stat,
                'p_value': p_value,
                'significance': p_to_stars(p_value)
            })
    return pd.DataFrame(rows)


def run_profiles(path,out):
    frame=read_table(path)
    check_ids(frame)
    if "cluster" not in frame or frame.cluster.isna().any() or frame.cluster.nunique()!=3:
        raise ValueError("Three complete cluster labels are required.")
    clinical=get_default_clinical_vars()
    if not set(clinical).issubset(frame):
        raise ValueError("The nineteen clinical and quantitative imaging fields are required.")
    result=compute_clinical_comparison_table(frame,clinical)
    pairs=compute_pairwise_mwu(frame,clinical)
    output=new_output(out,[path])
    result.to_csv(output/"clinical_global.csv",index=False)
    pairs.to_csv(output/"clinical_pairwise_continuous.csv",index=False)
    write_json(output/"analysis.json",dict(input_sha256=sha256(path),n=len(frame),
        categorical_test="Chi-square",continuous_test="Kruskal-Wallis",
        pairwise_test="Unadjusted two-sided Mann-Whitney U, exploratory", 
        unavailable_tests=int(result.p_value.isna().sum())))
    return result

