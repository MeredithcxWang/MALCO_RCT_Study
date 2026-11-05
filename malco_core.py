"""Shared schema and artifact contract for the public reproducibility release."""
from pathlib import Path
import json
import sys
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
SCHEMA = json.loads((ROOT / 'data/schema.json').read_text(encoding='utf-8'))
FEATURES, TASKS = SCHEMA['features'], SCHEMA['tasks']

def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def task_name(value):
    value = SCHEMA['aliases'].get(value, value)
    if value not in TASKS:
        raise ValueError(f'Unknown task {value}; expected one of {list(TASKS)}')
    return value

def read_table(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f'Input file does not exist: {path}')
    if path.suffix.lower() == '.csv': return pd.read_csv(path)
    if path.suffix.lower() == '.xlsx': return pd.read_excel(path)
    raise ValueError('Supported input formats: .csv, .xlsx')

def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.csv': df.to_csv(path, index=False)
    else: df.to_excel(path, index=False)

def feature_frame(df):
    missing = [name for name in FEATURES if name not in df]
    if missing: raise ValueError(f'Missing required feature columns: {missing}')
    frame = df.loc[:, FEATURES].apply(pd.to_numeric, errors='raise')
    if np.isinf(frame.to_numpy(dtype=float)).any(): raise ValueError('Features contain infinite values')
    if frame.isna().all(axis=1).any(): raise ValueError('A patient row has no observed features')
    return frame

def outcome(df, task):
    if task not in df: raise ValueError(f'Missing outcome column: {task}')
    y = pd.to_numeric(df[task], errors='raise')
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError(f'{task} must contain both binary classes, with no missing outcomes')
    return y.astype(int)

def load_artifact(models_dir, task):
    from xgboost import XGBClassifier
    task = task_name(task)
    folder = Path(models_dir) / task
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if meta.get('schema_version') != 1 or meta.get('features') != FEATURES or meta.get('task') != task:
        raise ValueError(f'Artifact schema mismatch for {task}')
    if meta.get('preprocessing') != 'none_xgboost_native_missing':
        raise ValueError('Unsupported preprocessing: do not attach an arbitrary scaler to a model')
    model = XGBClassifier()
    model.load_model(bytearray((folder / 'model.json').read_bytes()))
    if model.get_booster().feature_names != FEATURES: raise ValueError('Stored model feature schema mismatch')
    if not 0 < meta['threshold'] < 1: raise ValueError('Invalid model threshold')
    return model, meta

def prediction_table(df, task, probabilities, threshold):
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Invalid predicted probabilities')
    result = df.copy()
    if task in df: result['True_Label'] = df[task].to_numpy()
    result['Predicted_Probability'] = probabilities
    result['Predicted_Class'] = (probabilities >= threshold).astype(int)
    return result

def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
