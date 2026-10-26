"""Patient-aligned fusion of organ and clinical features."""
import re
import numpy as np
import pandas as pd
from .io import read_table, check_ids, binary_labels, new_output, write_json, sha256

ORGANS = ('Liver', 'Spleen', 'Muscle')
SPLITS = ('train', 'internal', 'external')
METADATA = ['imageName', 'Group', 'cohort']
FORBIDDEN = {'imagename', 'group', 'cohort', 'label', 'cluster', 'event', 'time_days',
             'death', 'mortality', 'rad_score', 'p_aclf', 'y_true'}


def validate_predictor_names(names):
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError('Provide a nonempty JSON array of predictor names.')
    if len(names) != len(set(names)) or FORBIDDEN.intersection(x.lower() for x in names):
        raise ValueError('Predictor names must be unique and exclude identifiers/outcomes.')


def validate_cohort_table(frame):
    check_ids(frame)
    if not set(METADATA).issubset(frame):
        raise ValueError('Required metadata: imageName, Group, cohort.')
    if frame.cohort.isna().any() or set(frame.cohort) != set(SPLITS):
        raise ValueError('Expected cohort labels: train, internal and external.')
    for split in SPLITS:
        binary_labels(frame.loc[frame.cohort == split, 'Group'])


def canonical_organ_table(frame, organ, dimensions=2048):
    """Map f_0000 and Organ_f_0 feature names to the organ-specific schema."""
    if organ not in ORGANS:
        raise ValueError('Unknown organ.')
    validate_cohort_table(frame)
    rename = {}
    for name in frame:
        match = re.fullmatch(rf'(?:{organ}_)?f_(\d+)', str(name))
        if match:
            rename[name] = f'{organ}_f_{int(match.group(1))}'
    expected = [f'{organ}_f_{i}' for i in range(dimensions)]
    if len(rename) != dimensions or len(set(rename.values())) != dimensions or set(rename.values()) != set(expected):
        raise ValueError(f'{organ}: expected exactly {dimensions} uniquely indexed deep features.')
    table = frame[METADATA + list(rename)].rename(columns=rename)
    table[expected] = table[expected].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(table[expected].to_numpy(dtype=float)).all():
        raise ValueError('Extracted embeddings must be finite; investigate missing extraction outputs.')
    return table[METADATA + expected]


def aligned_fusion(organ_frames, clinical, clinical_features, dimensions=2048):
    validate_predictor_names(clinical_features)
    if set(organ_frames) != set(ORGANS):
        raise ValueError('All three organ tables are required.')
    tables = {o: canonical_organ_table(organ_frames[o], o, dimensions) for o in ORGANS}
    validate_cohort_table(clinical)
    if not set(clinical_features).issubset(clinical):
        raise ValueError('The explicit clinical/quantitative-CT candidate list is incomplete.')
    if any(re.fullmatch(r'(Liver|Spleen|Muscle)_f_\d+', x) for x in clinical_features):
        raise ValueError('Clinical/quantitative-CT names must not duplicate deep feature names.')
    tables['ClinicalMP'] = clinical[METADATA + clinical_features].copy()
    numeric = tables['ClinicalMP'][clinical_features].apply(pd.to_numeric, errors='raise')
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError('Infinite clinical values must be resolved before fusion.')
    tables['ClinicalMP'][clinical_features] = numeric
    base = tables['Liver'][METADATA].copy().reset_index(drop=True)
    ids = base.imageName.astype(str)
    blocks = {}
    for name, table in tables.items():
        if set(table.imageName.astype(str)) != set(ids):
            raise ValueError(f'{name}: patient identifiers differ between input tables.')
        table = table.set_index(table.imageName.astype(str)).loc[ids].reset_index(drop=True)
        if not np.array_equal(table.Group, base.Group) or not np.array_equal(table.cohort, base.cohort):
            raise ValueError(f'{name}: conflicting outcome or cohort assignments.')
        blocks[name] = table.drop(columns=METADATA)
    features = {o: list(blocks[o]) for o in ORGANS}
    features['ClinicalMP'] = clinical_features
    features['LSMFusion'] = sum((features[o] for o in ORGANS), [])
    features['CLSMFusion'] = clinical_features + features['LSMFusion']
    all_features = pd.concat([base, blocks['ClinicalMP'], *(blocks[o] for o in ORGANS)], axis=1)
    return {
        name: all_features[METADATA + names].copy() for name,
        names in features.items()
    }, features


def run_fusion(liver, spleen, muscle, clinical, clinical_features, out):
    paths = {'Liver': liver, 'Spleen': spleen, 'Muscle': muscle}
    frames, features = aligned_fusion(
        {o: read_table(p) for o, p in paths.items()}, read_table(clinical), clinical_features
    )
    output = new_output(out, [*paths.values(), clinical])
    rows = []
    for name, frame in frames.items():
        folder = output / name
        folder.mkdir()
        write_json(folder / 'candidates.json', features[name])
        for split in SPLITS:
            table = frame.loc[frame.cohort == split].reset_index(drop=True)
            table.to_csv(folder / f'{split}.csv', index=False)
            rows.append(dict(model=name, cohort=split, n=len(table), events=int(table.Group.sum()),
                             candidate_features=len(features[name])))
    write_json(output / 'fusion.json', dict(
        inputs={o: sha256(p) for o, p in {**paths, 'ClinicalMP': clinical}.items()},
        configurations=rows, join='Exact patient-ID matching with outcome/cohort agreement',
        scope='Patient-aligned feature concatenation.'
    ))
    return pd.DataFrame(rows)
