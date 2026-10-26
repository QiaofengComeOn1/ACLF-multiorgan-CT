"""Single-organ agreement, block permutation and leave-one-organ-out analyses."""
import pandas as pd
from .io import read_table, new_output, check_ids, write_json, sha256
from .organ_contributions_core import (identify_organ_features, prepare_X, run_pca,
    evaluate_clustering_against_fusion, block_permutation_importance, optimal_label_mapping, contingency_df)


def run_contributions(training, out):
    frame = read_table(training)
    check_ids(frame)
    if len(frame) != 128 or frame.cluster.isna().any() or frame.cluster.nunique()!=3:
        raise ValueError('Expected the 128-case training cluster table with fixed labels.')
    output = new_output(out,[training])
    columns = identify_organ_features(frame.columns)
    if any(not block for block in columns):
        raise ValueError('Each organ must have an input feature block.')
    y = frame.cluster.to_numpy(dtype=int)
    blocks, rows = {}, []
    for organ, names in zip(['Liver','Spleen','Muscle'],columns):
        X = prepare_X(frame,names)
        res = evaluate_clustering_against_fusion(X,y,3)
        aligned, _ = optimal_label_mapping(y,res['labels_raw'])
        contingency_df(y,aligned).to_csv(output/f'{organ.lower()}_contingency.csv')
        rows.append(dict(
            Organ=organ,
            ARI=res['ARI'],
            NMI=res['NMI'],
            n_features=len(names),
            n_pca_components=res['Xp'].shape[1]
        ))
        blocks[organ.lower()] = run_pca(X,n_components=50,random_state=42)[0]
    pd.DataFrame(rows).to_csv(output/'single_organ_agreement.csv',index=False)
    permutation = block_permutation_importance(blocks,y,n_perm=200,seed=2026)
    permutation.to_csv(output/'block_permutation_importance.csv',index=False)
    liver,spleen,muscle = columns
    loo=[]
    for setting,names in [('All features',liver+spleen+muscle),('Without Liver',spleen+muscle),
                          ('Without Spleen',liver+muscle),('Without Muscle',liver+spleen)]:
        components = 50 if setting == 'All features' else .9
        res=evaluate_clustering_against_fusion(prepare_X(frame,names),y,3,n_components=components)
        loo.append(dict(
            Setting=setting,
            ARI=res['ARI'],
            NMI=res['NMI'],
            n_features=len(names),
            n_pca_components=res['Xp'].shape[1]
        ))
    pd.DataFrame(loo).to_csv(output/'leave_one_organ_out_ablation.csv',index=False)
    write_json(output/'analysis.json',dict(input_sha256=sha256(training),pca_components=50,
        leave_one_out_explained_variance=.9,permutations=200,
        permutation_seed=2026,kmeans_seed=42,cv_folds=5,
        scope='Descriptive organ-block dependence of training cluster assignments.'))
    return pd.DataFrame(rows)
