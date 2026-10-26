"""Two-stage fine-tuning of five-channel Inception-v3."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import timm
from .io import read_table, check_ids, binary_labels, new_output, write_json, sha256
from .imaging_core import (unwrap_state_dict, adapt_first_conv_in_state_dict,
                           standardize_batch)
from .imaging import prepare_image


def training_tensor(image_path, mask_path):
    return standardize_batch(prepare_image(image_path, mask_path))[0]


class FiveSliceDataset(Dataset):
    def __init__(self, frame, base):
        self.frame = frame.reset_index(drop=True)
        self.base = Path(base)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image, mask = Path(row.image), Path(row['mask'])
        image = image if image.is_absolute() else self.base / image
        mask = mask if mask.is_absolute() else self.base / mask
        return training_tensor(image, mask), torch.tensor(int(row.Group), dtype=torch.long)


def initialised_inception(weights):
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError('Local ImageNet checkpoint required.')
    model = timm.create_model('inception_v3', pretrained=False, in_chans=5, num_classes=2)
    state = unwrap_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    if not isinstance(state, dict):
        raise TypeError('Expected an author-controlled tensor state dictionary.')
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    state, _ = adapt_first_conv_in_state_dict(state, target_in_chans=5, method='mean')
    state = {k: v for k, v in state.items() if not k.startswith(('fc.', 'AuxLogits.'))}
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys
               if not k.startswith('fc.') and not k.endswith('num_batches_tracked')]
    if missing or result.unexpected_keys:
        raise ValueError('Pretrained weights do not fully initialise the backbone.')
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training/validation loss.')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
    return loss_sum / total, correct / total


def fine_tune(manifest, training_cohort, weights, organ, out, device='cpu', seed=42):
    frame, cohort = read_table(manifest), read_table(training_cohort)
    check_ids(frame)
    check_ids(cohort)
    if not {'image', 'mask', 'Group', 'cohort'}.issubset(frame):
        raise ValueError('Manifest requires imageName,image,mask,Group,cohort.')
    if 'cohort' not in cohort or not cohort.cohort.eq('train').all() or not frame.cohort.eq('train').all():
        raise ValueError('Backbone fitting is restricted to the prediction training cohort.')
    binary_labels(frame.Group)
    binary_labels(cohort.Group)
    if set(frame.imageName) != set(cohort.imageName):
        raise ValueError('Manifest must cover exactly the authoritative prediction training patients.')
    joined = frame[['imageName', 'Group']].merge(
        cohort[['imageName', 'Group']],
        on='imageName',
        validate='one_to_one'
    )
    if not np.array_equal(joined.Group_x, joined.Group_y):
        raise ValueError('Manifest labels disagree with the prediction training cohort.')
    base = Path(manifest).resolve().parent
    files = [Path(x) if Path(x).is_absolute() else base / x for c in ['image', 'mask'] for x in frame[c]]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError('Every training image and mask must exist.')
    train, validation = train_test_split(frame, test_size=.2, stratify=frame.Group, random_state=42)
    if len(train) < 2 or len(train) % 16 == 1:
        raise ValueError('BatchNorm training requires at least two samples per batch.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = initialised_inception(weights).to(device)
    output = new_output(out, [manifest, training_cohort, weights, *files])
    split_table = pd.concat([train.assign(backbone_split='development'),
                             validation.assign(backbone_split='validation')])
    split_table[['imageName', 'Group', 'backbone_split']].to_csv(output / 'backbone_split.csv', index=False)
    loaders = [DataLoader(FiveSliceDataset(f, base), batch_size=16, shuffle=shuffle,
                          num_workers=0, pin_memory=device.startswith('cuda'))
               for f, shuffle in [(train, True), (validation, False)]]
    criterion = nn.CrossEntropyLoss()
    history, best_acc = [], -1.0
    for stage, epochs, rate in [(1, 20, 1e-3), (2, 80, 1e-4)]:
        for parameter in model.parameters():
            parameter.requires_grad = stage == 2
        if stage == 1:
            for parameter in model.get_classifier().parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=rate)
        for epoch in range(epochs):
            train_loss, train_accuracy = run_epoch(model, loaders[0], criterion, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, loaders[1], criterion, device)
            history.append(dict(stage=stage, epoch=epoch, train_loss=train_loss,
                                train_accuracy=train_accuracy, validation_loss=val_loss,
                                validation_accuracy=val_accuracy))
            pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
            if stage == 2 and val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save(model.state_dict(), output / 'inception_5ch.pth')
    write_json(output / 'backbone_training.json', dict(
        architecture='inception_v3', organ=organ, manifest_sha256=sha256(manifest),
        training_cohort_sha256=sha256(training_cohort), initial_weights_sha256=sha256(weights),
        final_weights_sha256=sha256(output / 'inception_5ch.pth'), n=len(frame),
        development_n=len(train), validation_n=len(validation), seed=seed, split_seed=42,
        batch_size=16, epochs=[20, 80], learning_rates=[1e-3, 1e-4], device=device,
        input='Common masked bounding-box crop, resized to 224 and standardised per five-channel input.',
        changes='RNG seeds, input checks, relative paths and separate output directories.',
        provenance='Archived Inception-v3 training recipe.',
        determinism='cuDNN deterministic mode requested.'
    ))
    return dict(organ=organ, n=len(frame), best_validation_accuracy=best_acc)
