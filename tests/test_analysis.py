"""Input-validation and numerical tests."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aclf_ct.io import binary_labels, probabilities, selected_matrix, check_disjoint, new_output
from aclf_ct.metrics import find_best_threshold_by_youden, compute_auc_ci, delong_pvalue, calibration_curve_stats
from aclf_ct.evaluation import brier_interval
from aclf_ct.survival_core import clean_base_survival_df, add_endpoint_columns
from aclf_ct.clustering_core import preprocess_fit_transform, train_rf_and_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class AnalysisTests(unittest.TestCase):
    def test_invalid_label_and_probability_fail(self):
        for labels in ([0,1,np.nan],[0,2],[1,1]):
            with self.assertRaises(ValueError): binary_labels(labels)
        for probs in ([0,.2,1.1],[0,np.nan,.8]):
            with self.assertRaises(ValueError): probabilities(probs,3)

    def test_missing_test_features_are_not_imputed(self):
        frame=pd.DataFrame({'imageName':['a','b'],'Group':[0,1],'x':[1,np.nan]})
        with self.assertRaises(ValueError): selected_matrix(frame,['x'])

    def test_cohort_overlap_and_duplicate_detection(self):
        a=pd.DataFrame({'imageName':['a','b']})
        b=pd.DataFrame({'imageName':['b','c']})
        with self.assertRaises(ValueError): check_disjoint([a,b])
        with self.assertRaises(ValueError): check_disjoint([pd.DataFrame({'imageName':['a','a']})])

    def test_output_cannot_overwrite_or_write_under_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileExistsError): new_output(root)
            with self.assertRaises(ValueError): new_output(root/'child',[root])

    def test_training_imputation_and_scaling_only(self):
        a=pd.DataFrame({'x':[0.,2.,4.,np.nan]})
        b=pd.DataFrame({'x':[np.nan,100.]})
        _,z,imp,scale=preprocess_fit_transform(a,b)
        self.assertEqual(float(imp.statistics_[0]),2.)
        self.assertAlmostEqual(float(z[0,0]),0.)
        self.assertAlmostEqual(float(scale.mean_[0]),2.)

    def test_rf_transfer_is_not_refitted_on_full_training(self):
        rng=np.random.RandomState(17)
        X=rng.normal(size=(90,4)); y=np.repeat([0,1,2],30);Z=rng.normal(size=(10,4))
        model,_,_,labels,probs=train_rf_and_predict(X,y,Z)
        tr,te,yt,ye=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
        expected=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_split=2,
                    min_samples_leaf=1,class_weight='balanced',random_state=42,n_jobs=-1).fit(tr,yt)
        np.testing.assert_allclose(probs,expected.predict_proba(Z),rtol=0,atol=0)
        np.testing.assert_array_equal(labels,model.predict(Z))

    def test_missing_followup_and_endpoint_boundaries(self):
        frame=pd.DataFrame({'time_days':[28.,28.1,90.,np.nan,10.],
                            'event':[1,1,1,1,np.nan],'cluster':[0,1,2,0,1]})
        clean=add_endpoint_columns(clean_base_survival_df(frame))
        self.assertEqual(len(clean),3)
        self.assertEqual(clean.event_28.tolist(),[1,0,0])
        self.assertEqual(clean.event_90.tolist(),[1,1,1])
        self.assertEqual(clean.time_28.tolist(),[28.,28.,28.])

    def test_brier_matches_revision_bootstrap(self):
        y=np.array([0,1,1,0,1]);p=np.array([.1,.8,.6,.4,.9])
        score,lo,hi=brier_interval(y,p)
        squared=(p-y)**2
        rng=np.random.default_rng(20260915)
        boot=squared[rng.integers(0,len(y),size=(2000,len(y)))].mean(axis=1)
        self.assertEqual(score,float(squared.mean()))
        np.testing.assert_array_equal([lo,hi],np.quantile(boot,[.025,.975]))

    def test_auc_perfect_ranking_and_ties(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.2,.8,.8])
        auc,lo,hi=compute_auc_ci(y,p,n_bootstraps=40)
        self.assertEqual((auc,lo,hi),(1.,1.,1.))
        threshold,_=find_best_threshold_by_youden(y,p)
        self.assertEqual(threshold,.8)

    def test_calibration_keeps_boundary_probabilities(self):
        pred,observed,_,_=calibration_curve_stats(np.array([0,1]),np.array([0.,1.]))
        self.assertEqual(pred[0],0.)
        self.assertEqual(pred[-1],1.)
        self.assertEqual(observed[-1],1.)

    def test_delong_symmetric(self):
        y=np.array([0,1,0,1,0,1,0,1])
        a=np.array([.1,.6,.4,.3,.2,.8,.5,.7]);b=np.array([.3,.4,.2,.9,.1,.6,.8,.7])
        self.assertAlmostEqual(delong_pvalue(y,a,b),delong_pvalue(y,b,a))


if __name__=='__main__':
    unittest.main()
