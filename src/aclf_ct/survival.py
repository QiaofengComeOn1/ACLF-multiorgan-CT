"""Endpoint-specific survival analysis and follow-up accounting."""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from .io import read_table, new_output, write_json, sha256, check_disjoint
from .survival_core import clean_base_survival_df, add_endpoint_columns, fit_endpoint_cox_reference, get_endpoint_specs


def analyse_cohort(frame, cohort):
    for column in ('time_days','event','cluster'):
        if column not in frame:
            raise ValueError(f'Missing {column}.')
        values = pd.to_numeric(frame[column], errors='raise')
        if np.isinf(values).any():
            raise ValueError(f'Infinite {column} values are not permitted.')
    if not frame['event'].dropna().isin([0,1]).all() or (frame['time_days'].dropna() < 0).any():
        raise ValueError('Invalid follow-up time or event status.')
    clusters = sorted(frame['cluster'].dropna().unique())
    if clusters not in ([0,1,2], [1,2,3]):
        raise ValueError('Expected three fixed cluster labels (0/1/2 or 1/2/3).')
    complete = add_endpoint_columns(clean_base_survival_df(frame, cohort))
    cox = fit_endpoint_cox_reference(complete, cohort_name=cohort)
    if not cox['model_status'].eq('fitted').all():
        raise RuntimeError('Unpenalised Cox model failed to converge.')
    global_rows, km_rows = [], []
    for spec in get_endpoint_specs():
        t, e = spec['time_col'], spec['event_col']
        p = multivariate_logrank_test(complete[t], complete.cluster, complete[e]).p_value
        global_rows.append(dict(cohort=cohort, endpoint=spec['endpoint_short'], n=len(complete),
                                events=int(complete[e].sum()), global_logrank_p=float(p)))
        for label, group in complete.groupby('cluster', sort=True):
            km = KaplanMeierFitter().fit(group[t], group[e])
            for time, row in km.survival_function_.iterrows():
                km_rows.append(dict(cohort=cohort, endpoint=spec['endpoint_short'], cluster=int(label),
                                    time=float(time), survival=float(row.iloc[0])))
    missing = frame[['time_days','event','cluster']].isna()
    audit = dict(cohort=cohort, input_n=len(frame), analysed_n=len(complete),
                 excluded_missing_n=int(missing.any(axis=1).sum()),
                 missing_time=int(missing.time_days.sum()), missing_event=int(missing.event.sum()),
                 missing_both_time_and_event=int((missing.time_days & missing.event).sum()),
                 zero_time_clipped_to_1e_6=int(frame.time_days.eq(0).sum()))
    return cox, pd.DataFrame(global_rows), pd.DataFrame(km_rows), audit


def run_survival(training, external, out):
    frames = [read_table(training), read_table(external)]
    check_disjoint(frames)
    results = [analyse_cohort(f, c) for f,c in zip(frames, ['Training','External'])]
    output = new_output(out, [training, external])
    for index, name in enumerate(['cox_contrasts.csv', 'global_logrank.csv', 'km_curves.csv']):
        pd.concat([r[index] for r in results], ignore_index=True).to_csv(output/name, index=False)
    write_json(output/'missingness.json', [r[3] for r in results])
    write_json(
        output/'input_manifest.json',
        {'training':sha256(training), 'external':sha256(external)}
    )
    return pd.concat([r[1] for r in results], ignore_index=True)
