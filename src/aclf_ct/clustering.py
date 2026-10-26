"""Training-cohort consensus clustering and external cluster assignment."""
import joblib
import numpy as np
import pandas as pd
from .io import read_table, new_output, check_disjoint, sha256, write_json
from .clustering_core import (preprocess_fit_transform, consensus_clustering_kmeans,
                              cdf_area_from_consensus, reorder_labels_by_risk, train_rf_and_predict)


def deep_feature_columns(frame):
    columns = sorted(c for c in frame if c.startswith(('Liver_f_', 'Spleen_f_', 'Muscle_f_')))
    counts = {organ: sum(c.startswith(organ+'_f_') for c in columns) for organ in ('Liver','Spleen','Muscle')}
    if any(n != 2048 for n in counts.values()):
        raise ValueError('Primary clustering requires the complete 2048-feature block from each organ.')
    return columns


def run_clustering(training, external, out):
    train, test = read_table(training), read_table(external)
    check_disjoint([train,test])
    if len(train) != 128 or len(test) != 72:
        raise ValueError('The saved clustering profile requires 128 training and 72 external ACLF cases.')
    features = deep_feature_columns(train)
    if set(features) != set(deep_feature_columns(test)):
        raise ValueError('Organ feature sets differ between cohorts.')
    for frame in (train, test):
        if 'Group' in frame and not frame.Group.eq(1).all():
            raise ValueError('Clustering must contain ACLF cases only.')
    X, Z, imputer, scaler = preprocess_fit_transform(train[features], test[features])
    if X.shape[1] != 6144 or not np.isfinite(X).all() or not np.isfinite(Z).all():
        raise ValueError('Expected 6144 finite deep features per patient.')
    output = new_output(out,[training,external])
    consensus = consensus_clustering_kmeans(X, range(2,7), n_repeats=1000, p_item=.8, random_state=42)
    selection = pd.DataFrame([dict(k=k, cdf_area=cdf_area_from_consensus(r['consensus_matrix'])[0])
                              for k,r in consensus.items()]).sort_values('k')
    selection['delta_area'] = selection.cdf_area.diff()
    selection['relative_delta'] = selection.delta_area/selection.cdf_area.shift(1)
    best_k = int(selection.dropna(subset=['delta_area']).sort_values(['delta_area','k'],ascending=[False,True]).iloc[0]['k'])
    selection.to_csv(output/'consensus_k_selection.csv',index=False)
    if best_k != 3:
        raise RuntimeError('Selected cluster count differs from the configured value of three.')
    labels = consensus[best_k]['final_labels']
    
    
    
    if not {'time_days','event'}.issubset(train):
        raise ValueError('Training follow-up is required for mortality-based cluster numbering.')
    display_event = ((train.event == 1) & (train.time_days <= 28)).astype(int)
    labels, mapping = reorder_labels_by_risk(labels, display_event)
    classifier, accuracy, confusion, transferred, _ = train_rf_and_predict(X,labels,Z,random_state=42)
    train = train.copy(); test = test.copy()
    train['cluster'] = labels; test['cluster'] = transferred
    train.to_csv(output/'training_with_cluster.csv',index=False)
    test.to_csv(output/'external_with_cluster.csv',index=False)
    joblib.dump(
        dict(imputer=imputer,scaler=scaler,classifier=classifier,features=features),
        output/'cluster_transfer.joblib'
    )
    for k, value in consensus.items():
        np.save(output/f'consensus_k{k}.npy',value['consensus_matrix'])
    write_json(output/'analysis.json',dict(k=best_k,n_train=len(train),n_external=len(test),n_features=len(features),
        repeats=1000,seed=42,subsample_fraction=.8,pca_used=False,rf_trees=500,rf_training_fraction=.8,
        rf_refitted_on_all_training=False,rf_holdout_accuracy=float(accuracy),rf_holdout_confusion=confusion.tolist(),
        label_mapping={str(k):int(v) for k,v in mapping.items()},
        numbering_uses_training_28_day_outcome=True,numbering_missing_followup_treated_as_zero=True,
        training_sha256=sha256(training),external_sha256=sha256(external)))
    return pd.DataFrame({
        'Training':train.cluster.value_counts().sort_index(),
        'External':test.cluster.value_counts().sort_index()
    })
