"""Inception-v3 extraction from already prepared five-slice image/mask pairs."""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import timm
from .io import read_table, new_output, check_ids, write_json, sha256
from .fusion import validate_cohort_table, ORGANS
from .imaging_core import (crop_img_and_mask, preprocess_input, np_to_torch_5ch, resize_tensor,
    standardize_batch, unwrap_state_dict, adapt_first_conv_in_state_dict, filter_state_dict_for_model, extract_embedding)


def load_inception(weights, device='cpu', classification=False):
    """Load a complete five-channel Inception-v3 checkpoint."""
    weights=Path(weights)
    if not weights.is_file():
        raise FileNotFoundError('A local author-controlled checkpoint is required.')
    model=timm.create_model('inception_v3',pretrained=False,in_chans=5,
                            num_classes=2 if classification else 0,global_pool='avg')
    state=unwrap_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    if not isinstance(state,dict):
        raise TypeError('Expected a tensor state dictionary.')
    state,_=adapt_first_conv_in_state_dict(state,target_in_chans=5,method='mean')
    retained,_=filter_state_dict_for_model(state,model)
    result=model.load_state_dict(retained,strict=False)
    missing=[k for k in result.missing_keys if not k.endswith('num_batches_tracked')]
    if missing or result.unexpected_keys:
        raise ValueError('Checkpoint does not fully initialize the requested model.')
    return model.eval().to(device)


def prepare_image(image_path,mask_path):
    image_obj,mask_obj=nib.load(image_path),nib.load(mask_path)
    image,mask=image_obj.get_fdata(),mask_obj.get_fdata()
    if image.ndim!=3 or image.shape[-1]!=5 or image.shape!=mask.shape:
        raise ValueError('Each packed image and mask must share shape H x W x 5.')
    if not np.allclose(image_obj.affine,mask_obj.affine) or not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError('Image/mask geometry or finite-value checks failed.')
    if not (mask>0).any():
        raise ValueError('Empty organ mask.')
    image_crop,mask_crop,_=crop_img_and_mask(image,mask,margin=0)
    masked=preprocess_input(image_crop,mask_crop,use_mask_for_model=True)
    return resize_tensor(np_to_torch_5ch(masked),224)


def extract_features(manifest,weights,organ,out,device='cpu'):
    frame=read_table(manifest)
    check_ids(frame)
    if not {'image','mask','Group','cohort'}.issubset(frame):
        raise ValueError('Manifest columns: imageName,image,mask,Group,cohort.')
    validate_cohort_table(frame)
    if organ not in ORGANS:
        raise ValueError('Unknown organ.')
    parent=Path(manifest).resolve().parent
    paths=[]
    for row in frame.itertuples():
        image,mask=Path(row.image),Path(row.mask)
        paths.append((
            image if image.is_absolute() else parent/image,
            mask if mask.is_absolute() else parent/mask
        ))
    output=new_output(out,[manifest,weights]+[p for pair in paths for p in pair])
    model=load_inception(weights,device)
    embeddings=[]
    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision('high')
    with torch.no_grad():
        for start in range(0,len(frame),8):
            tensor=torch.cat([prepare_image(a,b) for a,b in paths[start:start+8]],dim=0).to(device)
            tensor=standardize_batch(tensor)
            context=torch.autocast('cuda') if device.startswith('cuda') else nullcontext()
            with context:
                features=extract_embedding(model,tensor)
            if features.shape[1]!=2048 or not torch.isfinite(features).all():
                raise ValueError('Expected 2048 finite Inception features.')
            embeddings.append(features.cpu().numpy())
    matrix=np.vstack(embeddings)
    table=pd.DataFrame(matrix,columns=[f'{organ}_f_{i}' for i in range(2048)])
    table=pd.concat([frame[['imageName','Group','cohort']].reset_index(drop=True),table],axis=1)
    table.to_csv(output/'features.csv',index=False)
    write_json(output/'extraction.json',dict(manifest_sha256=sha256(manifest),weights_sha256=sha256(weights),
        n=len(frame),organ=organ,features=2048,device=device,batch_size=8,image_size=224,
        margin_pixels=0,amp=device.startswith('cuda'),scope='Prepacked five-slice image and mask inputs.'))
    return dict(n=len(frame),features=2048,organ=organ)


def attribute_images(config_path, out, device='cpu', trusted=False):
    import json
    import joblib
    from .image_attribution_core import (
        FusionImageModel, ScoreCAM, occlusion_sensitivity, compute_shap_for_single_case
    )

    if not trusted:
        raise ValueError('Loading fitted model files requires --trust-model-files.')
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    organs = ('Liver', 'Spleen', 'Muscle')
    for key in ['weights', 'images', 'background']:
        if set(config[key]) != set(organs):
            raise ValueError(f'{key} must contain Liver, Spleen and Muscle.')
    sources = [config_path]

    def resolve(value):
        path = Path(value)
        path = path if path.is_absolute() else config_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(path)
        return path

    model_path = resolve(config['fusion_model'])
    preprocessing_path = resolve(config['preprocessor'])
    weight_paths = {organ: resolve(config['weights'][organ]) for organ in organs}
    image_paths = {organ: (resolve(config['images'][organ]['image']),
                           resolve(config['images'][organ]['mask'])) for organ in organs}
    background_paths = {}
    for organ in organs:
        cases = config['background'][organ]
        if not isinstance(cases, list) or not cases:
            raise ValueError('Each organ requires at least one background image/mask pair.')
        background_paths[organ] = [(resolve(case['image']), resolve(case['mask'])) for case in cases]
    predictor = joblib.load(model_path)
    preprocessing = joblib.load(preprocessing_path)
    if isinstance(preprocessing, dict):
        scaler = preprocessing['scaler']
        features = preprocessing['features']
        if 'features' in config and config['features'] != features:
            raise ValueError('Configured features disagree with the saved preprocessing bundle.')
    else:
        scaler = preprocessing
        features = config['features']
    backbones = {organ: load_inception(weight_paths[organ], device) for organ in organs}
    inputs = {organ: standardize_batch(prepare_image(*image_paths[organ])).to(device)
              for organ in organs}
    output = new_output(out, sources)
    predictions = {}
    for organ in organs:
        model = FusionImageModel(backbones, predictor, scaler, features, inputs, organ)
        target_layer = model.backbone.get_submodule(config.get('target_layer', 'Mixed_7c'))
        x = inputs[organ]
        scorecam = ScoreCAM(model, target_layer, max_channels=64, batch_size=16)
        try:
            cam, _, _, probability, _ = scorecam(x, target_class=1, out_size=(224, 224))
        finally:
            scorecam.remove_hooks()
        occlusion = occlusion_sensitivity(
            model, x, target_class=1, out_size=(224, 224), patch_size=24, stride=12,
            fill_mode='zero'
        )[0]
        background = torch.cat([
            standardize_batch(prepare_image(image, mask))
            for image, mask in background_paths[organ]
        ]).to(device)
        shap_map = compute_shap_for_single_case(model, background, x, target_class=1)[0]
        np.savez_compressed(output / f'{organ.lower()}_attribution.npz',
                            scorecam=cam, occlusion=occlusion, shap=shap_map)
        predictions[organ] = probability
    write_json(output / 'image_attribution.json', dict(
        model='LSMFusion', features=features, positive_class=1,
        fusion_model_sha256=sha256(model_path), preprocessor_sha256=sha256(preprocessing_path),
        weights_sha256={organ: sha256(path) for organ, path in weight_paths.items()},
        calibrated=hasattr(predictor, 'calibrated_classifiers_'),
        probabilities=predictions, fixed_other_organ_inputs=True,
        scorecam_channels=64, scorecam_batch_size=16, occlusion_window=24, occlusion_stride=12,
        device=device
    ))
    return dict(model='LSMFusion', organs=list(organs), probabilities=predictions)
