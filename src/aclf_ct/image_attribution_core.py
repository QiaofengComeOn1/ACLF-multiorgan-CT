"""Image-level attribution for fitted neural and fusion predictors."""

import numpy as np
import torch
import torch.nn.functional as F
from .imaging_core import resize_tensor, standardize_batch


class FusionImageModel(torch.nn.Module):
    expects_standardized_input = True

    def __init__(self, backbones, predictor, scaler, features, inputs, organ):
        super().__init__()
        from sklearn.linear_model import LogisticRegression

        organs = ('Liver', 'Spleen', 'Muscle')
        if organ not in organs or set(backbones) != set(organs) or set(inputs) != set(organs):
            raise ValueError('Fusion attribution requires all three organ inputs and backbones.')
        if not features or len(set(features)) != len(features):
            raise ValueError('An ordered, unique fusion feature list is required.')
        self.feature_indices = []
        for name in features:
            parts = name.split('_f_')
            if len(parts) != 2 or parts[0] not in organs or not parts[1].isdigit():
                raise ValueError('LSMFusion attribution accepts only organ deep features.')
            index = int(parts[1])
            if not 0 <= index < 2048:
                raise ValueError('Feature index is outside the 2048-dimensional embedding.')
            self.feature_indices.append((parts[0], index))
        if {name for name, _ in self.feature_indices} != set(organs):
            raise ValueError('The fusion feature set must represent all three organs.')
        if getattr(scaler, 'n_features_in_', None) != len(features):
            raise ValueError('Saved scaler dimensions do not match the fusion feature list.')
        if hasattr(scaler, 'feature_names_in_') and list(scaler.feature_names_in_) != list(features):
            raise ValueError('Saved scaler feature order differs from the fusion feature list.')
        mean = np.asarray(scaler.mean_, dtype=float)
        scale = np.asarray(scaler.scale_, dtype=float)
        if (mean.shape != (len(features),) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise ValueError('Finite training-fitted means and positive scales are required.')
        if not np.array_equal(predictor.classes_, [0, 1]):
            raise ValueError('The fusion classifier must use outcome classes 0 and 1.')
        self.calibrated = hasattr(predictor, 'calibrated_classifiers_')
        members = predictor.calibrated_classifiers_ if self.calibrated else [predictor]
        if not members:
            raise ValueError('The saved fusion classifier has no fitted estimators.')
        coefficients, intercepts, slopes, offsets = [], [], [], []
        for member in members:
            estimator = member.estimator if self.calibrated else member
            if (not isinstance(estimator, LogisticRegression) or not hasattr(estimator, 'coef_')
                    or np.asarray(estimator.coef_).shape != (1, len(features))
                    or not np.array_equal(estimator.classes_, [0, 1])):
                raise ValueError('Each fusion estimator must be a fitted binary logistic regression.')
            if (hasattr(estimator, 'feature_names_in_')
                    and list(estimator.feature_names_in_) != list(features)):
                raise ValueError('Classifier feature order differs from the saved preprocessing.')
            coefficients.append(estimator.coef_[0])
            intercepts.append(float(estimator.intercept_[0]))
            if self.calibrated:
                if len(member.calibrators) != 1:
                    raise ValueError('Expected one sigmoid calibrator per binary classifier.')
                calibrator = member.calibrators[0]
                if not hasattr(calibrator, 'a_') or not hasattr(calibrator, 'b_'):
                    raise ValueError('Only fitted sigmoid calibration is supported.')
                slopes.append(float(calibrator.a_))
                offsets.append(float(calibrator.b_))
        for name, values in [('feature_mean', mean), ('feature_scale', scale),
                             ('coefficients', coefficients), ('intercepts', intercepts),
                             ('calibration_slopes', slopes), ('calibration_offsets', offsets)]:
            array = np.asarray(values, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite fusion or calibration parameters.')
            self.register_buffer(name, torch.tensor(array, dtype=torch.float64))
        self.organ = organ
        self.backbone = backbones[organ].eval()
        for name in organs:
            image = inputs[name]
            if image.shape != (1, 5, 224, 224) or not torch.isfinite(image).all():
                raise ValueError('Each organ input must be a finite 1 x 5 x 224 x 224 tensor.')
            if name != organ:
                with torch.no_grad():
                    embedding = backbones[name].eval()(image)
                if embedding.shape != (1, 2048) or not torch.isfinite(embedding).all():
                    raise ValueError('Expected a finite 2048-dimensional organ embedding.')
                self.register_buffer(f'fixed_{name}', embedding.detach().clone())
        self.to(inputs[organ].device)
        self.eval()

    def forward(self, x):
        active = self.backbone(x)
        if active.shape != (x.shape[0], 2048) or not torch.isfinite(active).all():
            raise ValueError('Expected finite 2048-dimensional organ embeddings.')
        values = []
        for organ, index in self.feature_indices:
            if organ == self.organ:
                values.append(active[:, index])
            else:
                values.append(getattr(self, f'fixed_{organ}')[:, index].expand(x.shape[0]))
        features = torch.stack(values, dim=1).to(self.feature_mean.dtype)
        scaled = (features - self.feature_mean) / self.feature_scale
        decisions = scaled @ self.coefficients.T + self.intercepts
        if self.calibrated:
            probabilities = torch.sigmoid(-(
                decisions * self.calibration_slopes + self.calibration_offsets
            )).mean(dim=1)
            eps = torch.finfo(probabilities.dtype).eps
            probabilities = probabilities.clamp(eps, 1 - eps)
            score = torch.log(probabilities) - torch.log1p(-probabilities)
        else:
            score = decisions[:, 0]
        return torch.stack([torch.zeros_like(score), score], dim=1)


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self.handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _to_feature_map(self, tensor):
        if tensor is None:
            raise RuntimeError('Grad-CAM 尚未捕获到激活/梯度。')
        return feature_to_map(tensor, self.model)

    def __call__(self, x, target_class=None, out_size=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_class].item())
        pos_prob = float(probs[0, POSITIVE_CLASS].item())
        if target_class is None:
            target_class = pred_class
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)
        activations = self._to_feature_map(self.activations)
        gradients = self._to_feature_map(self.gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        if out_size is not None:
            cam = F.interpolate(cam, size=out_size, mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        cam = minmax01(cam)
        return (cam, pred_class, pred_prob, pos_prob, logits.detach().cpu().numpy())


class ScoreCAM:

    def __init__(self, model, target_layer, max_channels=64, batch_size=16, use_topk=True):
        self.model = model
        self.target_layer = target_layer
        self.max_channels = max_channels
        self.batch_size = batch_size
        self.use_topk = use_topk
        self.activations = None
        self.handles = []
        self.handles.append(self.target_layer.register_forward_hook(self._forward_hook))

    def _forward_hook(self, module, input, output):
        self.activations = output

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _to_feature_map(self, tensor):
        if tensor is None:
            raise RuntimeError('Score-CAM 尚未捕获到激活。')
        return feature_to_map(tensor, self.model)

    @torch.inference_mode()
    def __call__(self, x, target_class=None, out_size=None):
        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_class].item())
        pos_prob = float(probs[0, POSITIVE_CLASS].item())
        if target_class is None:
            target_class = pred_class
        feat = self._to_feature_map(self.activations)
        feat = feat.detach()
        B, C, h, w = feat.shape
        assert B == 1, '当前 Score-CAM 仅支持 batch=1'
        up_feat = F.interpolate(
            feat,
            size=(x.shape[2], x.shape[3]),
            mode='bilinear',
            align_corners=False
        )[0]
        channel_scores = up_feat.view(C, -1).std(dim=1)
        if self.use_topk and C > self.max_channels:
            idx = torch.topk(channel_scores, k=self.max_channels, largest=True).indices
            up_feat = up_feat[idx]
            C = up_feat.shape[0]
        fmap_flat = up_feat.view(C, -1)
        fmap_min = fmap_flat.min(dim=1)[0].view(C, 1, 1)
        fmap_max = fmap_flat.max(dim=1)[0].view(C, 1, 1)
        denom = (fmap_max - fmap_min).clamp_min(1e-08)
        norm_maps = (up_feat - fmap_min) / denom
        weighted_scores = []
        x_base = x[0:1]
        for start in range(0, C, self.batch_size):
            end = min(start + self.batch_size, C)
            maps_batch = norm_maps[start:end]
            maps_batch = maps_batch.unsqueeze(1)
            x_batch = x_base * maps_batch
            logits_batch = self.model(x_batch)
            if isinstance(logits_batch, (list, tuple)):
                logits_batch = logits_batch[0]
            probs_batch = torch.softmax(logits_batch, dim=1)[:, target_class]
            weighted_scores.append(probs_batch)
        weights = torch.cat(weighted_scores, dim=0).view(C, 1, 1)
        cam = (weights * norm_maps).sum(dim=0, keepdim=True).unsqueeze(0)
        cam = F.relu(cam)
        if out_size is not None:
            cam = F.interpolate(cam, size=out_size, mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        cam = minmax01(cam)
        return (cam, pred_class, pred_prob, pos_prob, logits.detach().cpu().numpy())


@torch.inference_mode()
def occlusion_sensitivity(
    model,
    x,
    target_class=None,
    out_size=None,
    patch_size=24,
    stride=12,
    fill_mode='zero'
):
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs = torch.softmax(logits, dim=1)
    pred_class = int(torch.argmax(probs, dim=1).item())
    pred_prob = float(probs[0, pred_class].item())
    pos_prob = float(probs[0, POSITIVE_CLASS].item())
    if target_class is None:
        target_class = pred_class
    base_score = float(probs[0, target_class].item())
    _, _, H, W = x.shape
    occ_map = torch.zeros((1, 1, H, W), device=x.device, dtype=torch.float32)
    cnt_map = torch.zeros((1, 1, H, W), device=x.device, dtype=torch.float32)
    if fill_mode == 'zero':
        fill_value = 0.0
    elif fill_mode == 'mean':
        fill_value = float(x.mean().item())
    else:
        raise ValueError(f'未知 OCCLUSION_FILL_MODE: {fill_mode}')
    y_positions = list(range(0, max(H - patch_size + 1, 1), stride))
    x_positions = list(range(0, max(W - patch_size + 1, 1), stride))
    if len(y_positions) == 0 or y_positions[-1] != H - patch_size:
        y_positions.append(max(H - patch_size, 0))
    if len(x_positions) == 0 or x_positions[-1] != W - patch_size:
        x_positions.append(max(W - patch_size, 0))
    for y in y_positions:
        for x0 in x_positions:
            x_occ = x.clone()
            x_occ[:, :, y:y + patch_size, x0:x0 + patch_size] = fill_value
            logits_occ = model(x_occ)
            if isinstance(logits_occ, (list, tuple)):
                logits_occ = logits_occ[0]
            probs_occ = torch.softmax(logits_occ, dim=1)
            occ_score = float(probs_occ[0, target_class].item())
            drop = max(base_score - occ_score, 0.0)
            occ_map[:, :, y:y + patch_size, x0:x0 + patch_size] += drop
            cnt_map[:, :, y:y + patch_size, x0:x0 + patch_size] += 1.0
    occ_map = occ_map / cnt_map.clamp_min(1e-08)
    if out_size is not None:
        occ_map = F.interpolate(occ_map, size=out_size, mode='bilinear', align_corners=False)
    occ_map = occ_map[0, 0].detach().cpu().numpy().astype(np.float32)
    occ_map = minmax01(occ_map)
    return (occ_map, pred_class, pred_prob, pos_prob, logits.detach().cpu().numpy())


def compute_integrated_gradients(model, x, target_class=None, baseline_mode='zero', n_steps=64):
    model.eval()
    x = x.clone().to(next(model.parameters()).device)
    if not getattr(model, 'expects_standardized_input', False):
        x = standardize_batch(x)
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs, pred_class, pred_prob, pos_prob = softmax_probs(logits)
    if target_class is None:
        target_class = pred_class
    if baseline_mode == 'zero':
        baseline = torch.zeros_like(x)
    elif baseline_mode == 'mean':
        baseline = torch.ones_like(x) * x.mean()
    else:
        raise ValueError(f'未知 IG_BASELINE_MODE: {baseline_mode}')
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        inputs=x,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    attr_map = attributions[0].detach().cpu().numpy().astype(np.float32)
    return (attr_map, pred_class, pred_prob, pos_prob, logits.detach().cpu().numpy())


def minmax01(arr):
    arr = arr.astype(np.float32)
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-08:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def reshape_transform_for_vit(tensor, model):
    if tensor.ndim != 3:
        raise ValueError(f'ViT reshape_transform 输入维度应为 3，实际为 {tensor.ndim}')
    B, N, C = tensor.shape
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'grid_size'):
        gh, gw = model.patch_embed.grid_size
    elif int(math.sqrt(N - 1)) ** 2 == N - 1:
        gh = gw = int(math.sqrt(N - 1))
    elif int(math.sqrt(N)) ** 2 == N:
        gh = gw = int(math.sqrt(N))
    else:
        raise ValueError(f'无法从 token 数 {N} 推断 ViT 网格大小')
    if N == gh * gw + 1:
        tensor = tensor[:, 1:, :]
    elif N == gh * gw:
        pass
    elif int(math.sqrt(N - 1)) ** 2 == N - 1:
        s = int(math.sqrt(N - 1))
        tensor = tensor[:, 1:, :]
        gh = gw = s
    elif int(math.sqrt(N)) ** 2 == N:
        s = int(math.sqrt(N))
        gh = gw = s
    else:
        raise ValueError(f'token 数 {N} 与网格不匹配')
    tensor = tensor.reshape(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()
    return tensor


def feature_to_map(tensor, model):
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        return reshape_transform_for_vit(tensor, model)
    raise ValueError(f'不支持的特征维度: {tensor.ndim}')


def softmax_probs(logits):
    probs = torch.softmax(logits, dim=1)
    pred_class = int(torch.argmax(probs, dim=1).item())
    pred_prob = float(probs[0, pred_class].item())
    pos_prob = float(probs[0, POSITIVE_CLASS].item())
    return (probs, pred_class, pred_prob, pos_prob)

import math
from captum.attr import IntegratedGradients
POSITIVE_CLASS = 1

NUM_CLASSES = 2

def compute_shap_for_single_case(model, background_tensor, x, target_class=None):
    """Return a five-channel SHAP map from supported SHAP output layouts."""
    import shap
    model.eval()
    x = x.clone().to(next(model.parameters()).device)
    if not getattr(model, 'expects_standardized_input', False):
        x = standardize_batch(x)
    logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    probs, pred_class, pred_prob, pos_prob = softmax_probs(logits)
    if target_class is None:
        target_class = pred_class
    explainer = shap.GradientExplainer(model, background_tensor)
    shap_values = explainer.shap_values(x, rseed=0)
    if isinstance(shap_values, list):
        shap_map = shap_values[target_class]
        shap_map = np.asarray(shap_map, dtype=np.float32)
        if shap_map.ndim == 4 and shap_map.shape[0] == 1:
            shap_map = shap_map[0]
        if shap_map.ndim != 3:
            raise ValueError(f'SHAP(list) 输出维度异常: {shap_map.shape}，期望为 (5,H,W)')
        return (shap_map, pred_class, pred_prob, pos_prob, logits.detach().cpu().numpy())
    shap_values = np.asarray(shap_values, dtype=np.float32)
    if shap_values.ndim == 5 and shap_values.shape[0] == 1:
        shap_values = shap_values[0]
    if shap_values.ndim == 4 and shap_values.shape[-1] == NUM_CLASSES:
        shap_map = shap_values[..., target_class]
        return (
            shap_map.astype(np.float32),
            pred_class,
            pred_prob,
            pos_prob,
            logits.detach().cpu().numpy()
        )
    if shap_values.ndim == 4 and shap_values.shape[0] == 1:
        shap_map = shap_values[0]
        return (
            shap_map.astype(np.float32),
            pred_class,
            pred_prob,
            pos_prob,
            logits.detach().cpu().numpy()
        )
    if shap_values.ndim == 3:
        shap_map = shap_values
        return (
            shap_map.astype(np.float32),
            pred_class,
            pred_prob,
            pos_prob,
            logits.detach().cpu().numpy()
        )
    raise ValueError(f'SHAP 输出维度异常: {shap_values.shape}，无法解析为 (5,H,W)')
