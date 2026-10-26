"""Backbone preprocessing tests with synthetic arrays."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    import nibabel as nib
    import timm
    from aclf_ct.backbone_training import training_tensor, run_epoch
    from aclf_ct.imaging import prepare_image
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Optional imaging dependencies are not installed.')
class BackboneTests(unittest.TestCase):
    def test_tensor_geometry_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(3)
            image = rng.normal(size=(80, 80, 5)).astype(np.float32)
            mask = np.zeros_like(image)
            mask[25:50, 25:50] = 1
            ip, mp = root / 'image.nii.gz', root / 'mask.nii.gz'
            nib.save(nib.Nifti1Image(image, np.eye(4)), ip)
            nib.save(nib.Nifti1Image(mask, np.eye(4)), mp)
            tensor = training_tensor(ip, mp)
            self.assertEqual(tuple(tensor.shape), (5, 224, 224))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertAlmostEqual(float(tensor.mean()), 0., places=5)
            self.assertAlmostEqual(float(tensor.std()), 1., places=5)
            self.assertEqual(tuple(prepare_image(ip, mp).shape), (1, 5, 224, 224))
            nib.save(nib.Nifti1Image(mask, np.diag([2., 2., 2., 1.])), mp)
            with self.assertRaises(ValueError):
                training_tensor(ip, mp)
            model = torch.nn.Linear(3, 2)
            original = model.weight.detach().clone()
            batches = [(torch.ones(4, 3), torch.tensor([0, 0, 0, 1]))]
            loss, _ = run_epoch(model, batches, torch.nn.CrossEntropyLoss(), 'cpu',
                                torch.optim.Adam(model.parameters(), lr=.01))
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(original, model.weight))


if __name__ == '__main__':
    unittest.main()
