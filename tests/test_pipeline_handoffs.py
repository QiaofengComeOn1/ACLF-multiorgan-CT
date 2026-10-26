"""Synthetic-data tests for alignment and preprocessing interfaces."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aclf_ct.fusion import aligned_fusion, canonical_organ_table
from aclf_ct.selection import run_selection, historical_parameters
from aclf_ct.training import fit_model
from aclf_ct.prediction import predict_bundle
from aclf_ct.preprocessing import prepare_selected


class HandoffTests(unittest.TestCase):
    def fixture(self):
        base = pd.DataFrame(dict(imageName=[f'fake_{i}' for i in range(12)],
                                 Group=[0, 1] * 6,
                                 cohort=['train'] * 4 + ['internal'] * 4 + ['external'] * 4))
        organs = {o: base.assign(f_0000=np.arange(12) + v, f_0001=np.arange(12) + 100 + v)
                  for o, v in [('Liver', 0), ('Spleen', 20), ('Muscle', 40)]}
        return organs, base.assign(ALT=np.arange(12) * 3.)

    def test_fusion_aligns_shuffled_rows_and_preserves_values(self):
        organs, clinical = self.fixture()
        organs['Spleen'] = organs['Spleen'].sample(frac=1, random_state=9)
        clinical = clinical.iloc[::-1]
        tables, names = aligned_fusion(organs, clinical, ['ALT'], dimensions=2)
        np.testing.assert_array_equal(tables['CLSMFusion'].Spleen_f_0, np.arange(12) + 20)
        np.testing.assert_array_equal(tables['CLSMFusion'].ALT, np.arange(12) * 3.)
        self.assertEqual(len(names['LSMFusion']), 6)
        self.assertEqual(len(tables), 6)

    def test_fusion_rejects_conflict_missing_patient_and_ambiguous_feature(self):
        organs, clinical = self.fixture()
        bad = clinical.copy()
        bad.loc[0, 'Group'] = 1
        with self.assertRaises(ValueError):
            aligned_fusion(organs, bad, ['ALT'], 2)
        with self.assertRaises(ValueError):
            aligned_fusion(organs, clinical.iloc[1:], ['ALT'], 2)
        bad = organs['Liver'].assign(Liver_f_0=0)
        with self.assertRaises(ValueError):
            canonical_organ_table(bad, 'Liver', 2)
        with self.assertRaises(ValueError):
            aligned_fusion(organs, clinical, ['Group'], 2)

    def test_selection_training_prediction_and_isolation(self):
        
        
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'input'
            raw.mkdir()
            rng = np.random.default_rng(13)
            frames, paths = {}, {}
            for split, n, offset in [('train', 60, 0), ('internal', 20, 30), ('external', 20, 50)]:
                y = np.tile([0, 1], n // 2)
                frame = pd.DataFrame(dict(imageName=[f'fake_{split}_{i}' for i in range(n)],
                                          Group=y, cohort=split, x=y * 3 + rng.normal(size=n) + offset))
                frame.loc[0, 'x'] = np.nan
                frames[split] = frame
                paths[split] = raw / f'{split}.csv'
                frame.to_csv(paths[split], index=False)
            params = historical_parameters()
            params.update(Cs=np.array([.1, 1.]), logit_cv_folds=2, logit_n_jobs=1)
            selected = root / 'selected'
            with patch('aclf_ct.selection.historical_parameters', return_value=params), \
                 patch('aclf_ct.selection.select_features_one_fold', return_value=['x']) as fold:
                run_selection(
                    paths['train'],
                    ['x'],
                    .5,
                    selected,
                    paths['internal'],
                    paths['external']
                )
                self.assertEqual(fold.call_count, 30)
            transform = joblib.load(selected / 'preprocessor.joblib')
            median = frames['train'].x.median()
            self.assertAlmostEqual(transform['cleaner']['imputer'].statistics_[0], median)
            internal = pd.read_csv(selected / 'internal_scaled.csv')
            expected = transform['scaler'].transform(pd.DataFrame({'x': [median]}))[0, 0]
            self.assertAlmostEqual(internal.loc[0, 'x'], expected)
            bundle = root / 'trained'
            fit_model(selected / 'train_scaled.csv', 'LSMFusion', bundle, selection_dir=selected)
            result = predict_bundle(bundle, selected / 'internal_scaled.csv',
                                    selected / 'external_scaled.csv', root / 'predicted', True, bootstraps=20)
            self.assertEqual(len(result), 2)
            metadata = json.loads((bundle / 'training.json').read_text())
            self.assertEqual(metadata['mode'], 'new_selection')
            self.assertTrue((result.threshold == metadata['threshold']).all())
            with self.assertRaises(ValueError):
                predict_bundle(bundle, selected / 'train_scaled.csv', selected / 'external_scaled.csv',
                               root / 'overlap', True, bootstraps=20)
            with self.assertRaises(ValueError):
                predict_bundle(bundle, selected / 'internal_scaled.csv', selected / 'external_scaled.csv',
                               root / 'untrusted', False)
            with self.assertRaises(FileExistsError):
                predict_bundle(bundle, selected / 'internal_scaled.csv', selected / 'external_scaled.csv',
                               root / 'predicted', True, bootstraps=20)

    def test_selected_preprocessing_uses_training_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'inputs'
            inputs.mkdir()
            paths = {}
            for split, values in [
                ('train', [0., 2.]),
                ('internal', [100., 102.]),
                ('external', [-5., -3.])
            ]:
                paths[split] = inputs / f'{split}.csv'
                pd.DataFrame(dict(imageName=[f'fake_{split}_{i}' for i in range(2)],
                                  Group=[0, 1], cohort=split, x=values)).to_csv(paths[split], index=False)
            config = {'LSMFusion': dict(features=['x'], counts=dict(train=2, internal=2, external=2),
                                        events=dict(train=1, internal=1, external=1))}
            out = root / 'scaled'
            with patch('aclf_ct.preprocessing.profile', return_value=config):
                prepare_selected(
                    paths['train'],
                    paths['internal'],
                    paths['external'],
                    'LSMFusion',
                    out
                )
            np.testing.assert_allclose(pd.read_csv(out / 'internal_scaled.csv').x, [99., 101.])


if __name__ == '__main__':
    unittest.main()
