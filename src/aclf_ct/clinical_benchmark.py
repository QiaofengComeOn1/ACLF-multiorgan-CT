"""Discrimination analysis of recorded MELD scores."""
import numpy as np
import pandas as pd
from .io import read_table, binary_labels, new_output, write_json, sha256
from .metrics import compute_auc_ci


def run_meld(path,out):
    frame=read_table(path)
    if not {'cohort','Group','MELD'}.issubset(frame):
        raise ValueError('Expected cohort, Group and precomputed MELD columns.')
    rows=[]
    for value,name in [(1,'Training'),(2,'Internal'),(3,'External')]:
        original=frame.loc[frame.cohort==value,['Group','MELD']]
        data=original.dropna()
        y=binary_labels(data.Group)
        scores=data.MELD.to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite MELD values.')
        auc,lower,upper=compute_auc_ci(y,scores,n_bootstraps=1000,seed=2024)
        rows.append(dict(
            cohort=name,
            n=len(data),
            excluded_missing=len(original)-len(data),
            events=int(y.sum()),
            AUC=auc,
            AUC_low=lower,
            AUC_high=upper
        ))
    output=new_output(out,[path])
    pd.DataFrame(rows).to_csv(output/'meld_auc.csv',index=False)
    write_json(output/'analysis.json',dict(input_sha256=sha256(path),bootstrap=1000,seed=2024,
        threshold_metrics_omitted='Test-cohort cut-offs: exploratory.'))
    return pd.DataFrame(rows)
