"""CPU attribution tests with synthetic tensors."""
from pathlib import Path
import importlib.util
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
AVAILABLE=all(importlib.util.find_spec(name) for name in ['torch','timm','captum','shap'])


@unittest.skipUnless(AVAILABLE,'Optional imaging/attribution dependencies are not installed.')
class ImagingTests(unittest.TestCase):
    def test_preprocessing_and_attribution_are_finite(self):
        import numpy as np
        import torch
        from aclf_ct.imaging_core import np_to_torch_5ch,resize_tensor,standardize_batch
        from aclf_ct.image_attribution_core import (GradCAM,ScoreCAM,occlusion_sensitivity,
            compute_integrated_gradients,compute_shap_for_single_case)
        torch.manual_seed(4)
        x=standardize_batch(resize_tensor(
            np_to_torch_5ch(np.arange(16*16*5,dtype='float32').reshape(16,16,5)),
            16
        ))
        model=torch.nn.Sequential(torch.nn.Conv2d(5,4,3,padding=1),torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,2)).eval()
        for cls in (GradCAM,ScoreCAM):
            explainer=cls(model,model[0])
            try: result=explainer(x,target_class=1,out_size=(16,16))
            finally: explainer.remove_hooks()
            self.assertEqual(result[0].shape,(16,16))
            self.assertTrue(np.isfinite(result[0]).all())
        result=occlusion_sensitivity(model,x,target_class=1,patch_size=8,stride=8,out_size=(16,16))
        self.assertTrue(np.isfinite(result[0]).all())
        result=compute_integrated_gradients(model,x,target_class=1,n_steps=8)
        self.assertEqual(result[0].shape,(5,16,16))
        result=compute_shap_for_single_case(model,torch.cat([x,-x],dim=0),x,target_class=1)
        self.assertEqual(result[0].shape,(5,16,16))
        self.assertTrue(np.isfinite(result[0]).all())


if __name__=='__main__':
    unittest.main()
