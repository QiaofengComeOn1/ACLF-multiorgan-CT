"""Input validation and analysis file handling."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, dtype={'imageName': str, 'case_id': str})
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path, dtype={'imageName': str, 'case_id': str})
    raise ValueError('Expected a CSV or Excel table.')


def new_output(path, inputs=()):
    """Require a new output directory outside every input directory."""
    output = Path(path).resolve()
    for item in inputs:
        source = Path(item).resolve()
        if source.is_file():
            source = source.parent
        if output == source or source in output.parents:
            raise ValueError('Output must be outside the input directory.')
    output.mkdir(parents=True, exist_ok=False)
    return output


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8'
    )


def profile():
    return json.loads(Path(__file__).with_name('paper_profile.json').read_text(encoding='utf-8'))


def check_ids(frame, column='imageName'):
    if column not in frame or frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f'{column} must exist, be nonmissing and be unique within a cohort.')
    if frame[column].astype(str).str.strip().eq('').any():
        raise ValueError('Empty case identifiers are not permitted.')


def check_disjoint(frames):
    seen = set()
    for frame in frames:
        check_ids(frame)
        identifiers = set(frame['imageName'].astype(str))
        if identifiers & seen:
            raise ValueError('Case identifiers overlap between cohorts. Resolve before analysis.')
        seen |= identifiers


def binary_labels(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Outcome must be a nonempty binary vector without missing values.')
    if len(np.unique(y)) != 2:
        raise ValueError('Both outcome classes are required for these analyses.')
    return y.astype(int)


def probabilities(values, n):
    p = np.asarray(values, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite, in [0, 1] and aligned with outcomes.')
    return p


def selected_matrix(frame, features):
    check_ids(frame)
    y = binary_labels(frame['Group'])
    if len(set(features)) != len(features) or not set(features).issubset(frame.columns):
        raise ValueError('The ordered frozen feature set is incomplete or duplicated.')
    X = frame.loc[:, features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError('Selected feature matrices must contain finite values.')
    return X, y
