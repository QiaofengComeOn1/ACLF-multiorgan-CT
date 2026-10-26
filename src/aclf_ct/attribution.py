"""Feature-level SHAP analysis of the raw LSMFusion classifier."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from .io import read_table, selected_matrix, profile, new_output, write_json, sha256


def run_attribution(run,cohort,out,trusted=False):
    if not trusted:
        raise ValueError('Explicit trust is required for author-controlled model files.')
    run=Path(run)
    config=profile()['LSMFusion']
    table=read_table(run/'preprocess'/f'{cohort}_scaled.csv')
    X,_=selected_matrix(table,config['features'])
    model_path=run/'train'/'best_raw_model_LR.pkl'
    model=joblib.load(model_path)
    if X.shape[1]!=13:
        raise ValueError('The selected attribution target must be the 13-feature raw LSMFusion model.')
    
    
    rng=np.random.RandomState(0)
    background=X[rng.choice(len(X),size=min(100,len(X)),replace=False)]
    predict=lambda values:model.predict_proba(pd.DataFrame(values))[:,1]
    explainer=shap.Explainer(
        predict,
        shap.maskers.Independent(background),
        algorithm='permutation',
        seed=0
    )
    values=explainer(X)
    output=new_output(out,[run])
    np.save(output/'shap_values.npy',values.values)
    np.save(output/'base_values.npy',values.base_values)
    importance=np.abs(values.values).mean(axis=0)
    result=pd.DataFrame({'feature':config['features'],'mean_abs_shap':importance}).sort_values('mean_abs_shap',ascending=False)
    result.to_csv(output/'feature_importance.csv',index=False)
    groups=result.feature.str.split('_').str[0]
    result.groupby(groups).mean_abs_shap.sum().to_csv(output/'organ_importance.csv')
    pd.DataFrame(X,columns=config['features']).corr(method='spearman').to_csv(output/'spearman.csv')
    write_json(output/'attribution.json',dict(model_sha256=sha256(model_path),target='Raw LSMFusion probability',
        calibrated_model_explained=False,background_cohort=cohort,background_n=len(background),seed=0,
        reproduction='Permutation SHAP with seed 0 and cohort-specific background sampling.'))
    return result
