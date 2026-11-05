"""Evaluate a saved artifact. Missing inputs and failed evaluations are errors."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from malco_core import resolve_path, task_name, read_table, feature_frame, outcome, load_artifact, prediction_table, write_table, write_json

def evaluate(args):
    task = task_name(args.task)
    df = read_table(resolve_path(args.input))
    X, y = feature_frame(df), outcome(df, task)
    model, meta = load_artifact(resolve_path(args.models_dir), task)
    probs = model.predict_proba(X)[:, 1]
    directory = resolve_path(args.output_dir) / 'external' / task
    write_table(prediction_table(df, task, probs, meta['threshold']), directory / f'{task}_prediction_results.xlsx')
    score = float(roc_auc_score(y, probs))
    fpr, tpr, _ = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f'AUROC {score:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{task}: '+('synthetic demonstration' if meta['synthetic_demo'] else 'supplied evaluation cohort'))
    ax.legend()
    fig.savefig(directory / f'{task}_ROC.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    result = {'task':task, 'n':len(df), 'auroc':score, 'synthetic_model':meta['synthetic_demo'], 'completed':1, 'failed':0}
    write_json(result, directory / 'metrics.json')
    print(f'{task}: evaluated {len(df)} rows; AUROC={score:.4f}')
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--task', default='2yrM')
    p.add_argument('--models-dir', default='output/demo/models')
    p.add_argument('--output-dir', default='output/demo')
    try: evaluate(p.parse_args())
    except Exception as exc: p.exit(1, f'Evaluation failed: {exc}\n')

if __name__ == '__main__': main()
