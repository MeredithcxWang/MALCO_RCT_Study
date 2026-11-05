"""Train a selected task and save a reloadable, unscaled XGBoost artifact.
This public implementation does not establish historical trial model provenance.
"""
import argparse, hashlib, importlib.metadata
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from malco_core import ROOT, FEATURES, TASKS, resolve_path, read_table, feature_frame, outcome, task_name, write_table, write_json, prediction_table

def train(args):
    task, source = task_name(args.task), resolve_path(args.input)
    df = read_table(source)
    X, y = feature_frame(df), outcome(df, task)
    if not 0 < args.test_size < 1 or not 0 < args.threshold < 1:
        raise ValueError('test-size and threshold must be strictly between 0 and 1')
    train_ids, test_ids = train_test_split(np.arange(len(df)), test_size=args.test_size, random_state=args.seed, stratify=y)
    if args.cv < 2 or int(y.iloc[train_ids].value_counts().min()) < args.cv:
        raise ValueError('Each training class must have at least --cv observations')
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', missing=np.nan,
                          random_state=args.seed, n_jobs=1, tree_method='hist')
    demo_space = {'n_estimators':[30,60,100], 'max_depth':[2,3,4], 'learning_rate':[.03,.1,.2],
                  'subsample':[.8,1.0], 'colsample_bytree':[.8,1.0]}
    original_space = {'learning_rate':uniform(.01,.15), 'max_depth':randint(3,7),
        'colsample_bytree':uniform(.4,.6), 'colsample_bylevel':uniform(.4,.6), 'subsample':uniform(.3,.6),
        'n_estimators':randint(200,500), 'alpha':uniform(0,5), 'reg_lambda':uniform(.5,9.5),
        'gamma':uniform(0,5), 'min_child_weight':randint(1,10)}
    search = RandomizedSearchCV(model, demo_space if args.demo else original_space,
        n_iter=args.search_iterations, scoring='roc_auc', cv=StratifiedKFold(args.cv, shuffle=True, random_state=args.seed),
        random_state=args.seed, n_jobs=args.jobs, error_score='raise')
    search.fit(X.iloc[train_ids], y.iloc[train_ids])
    model = search.best_estimator_
    output = resolve_path(args.output_dir)
    folder = output / 'models' / task
    folder.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(X.iloc[test_ids])[:, 1]
    metadata = {'schema_version':1, 'task':task, 'model_name':TASKS[task], 'features':FEATURES,
        'preprocessing':'none_xgboost_native_missing', 'threshold':args.threshold,
        'threshold_source':'explicit CLI value; not a recovered clinical threshold', 'synthetic_demo':args.demo,
        'seed':args.seed, 'test_size':args.test_size, 'cv_folds':args.cv, 'search_iterations':args.search_iterations,
        'best_parameters':search.best_params_, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'train_rows':len(train_ids), 'test_rows':len(test_ids), 'cv_auroc':float(search.best_score_),
        'holdout_auroc':float(roc_auc_score(y.iloc[test_ids], probabilities)),
        'versions':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','xgboost')}}
    (folder / 'model.json').write_bytes(model.get_booster().save_raw(raw_format='json'))
    write_json(metadata, folder / 'metadata.json')
    write_table(prediction_table(df.iloc[test_ids], task, probabilities, args.threshold),
                output / 'internal' / task / f'{task}_prediction_results.xlsx')
    print(f'{task}: saved model and {len(test_ids)} holdout predictions; AUROC={metadata["holdout_auroc"]:.4f}')
    return metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/synthetic_sample.xlsx')
    p.add_argument('--task', default='2yrM')
    p.add_argument('--output-dir', default='output/demo')
    p.add_argument('--test-size', type=float, default=.2)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cv', type=int)
    p.add_argument('--search-iterations', type=int)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--demo', action='store_true', help='Label artifacts as synthetic demonstration models')
    args = p.parse_args()
    if resolve_path(args.input) == ROOT/'data/synthetic_sample.xlsx': args.demo = True
    args.cv = args.cv if args.cv is not None else (3 if args.demo else 10)
    args.search_iterations = args.search_iterations if args.search_iterations is not None else (2 if args.demo else 50)
    if args.search_iterations < 1 or args.jobs < 1: p.error('search-iterations and jobs must be positive')
    try: train(args)
    except Exception as exc: p.exit(1, f'Training failed: {exc}\n')

if __name__ == '__main__': main()
