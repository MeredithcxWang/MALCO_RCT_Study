"""Build all eight synthetic demonstration artifacts; fail on any incomplete stage."""
import argparse, importlib.util, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from malco_core import TASKS, FEATURES, read_table, write_table, resolve_path, write_json
from ml_prognostic_models.xgboost_training import train
from ml_prognostic_models.model_evaluation import evaluate
from argparse import Namespace

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='output/demo')
    args=p.parse_args()
    out=resolve_path(args.output_dir)
    started=time.monotonic()
    subprocess.run([sys.executable,str(ROOT/'generate_synthetic_data.py'),'--output-dir',str(out/'input')],check=True)
    metrics=[]
    for task in TASKS:
        train(Namespace(task=task,input=str(out/'input/train.xlsx'),output_dir=str(out),test_size=.2,
                        threshold=.5,seed=42,cv=3,search_iterations=2,jobs=1,demo=True))
        metrics.append(evaluate(Namespace(task=task,input=str(out/'input/evaluation.xlsx'),
                                        models_dir=str(out/'models'),output_dir=str(out))))
    df=read_table(out/'input/evaluation.xlsx')
    write_table(df.loc[:,FEATURES].iloc[:1],out/'input/patient.csv')
    subprocess.run([sys.executable,str(ROOT/'malco_framework/run_models.py'),'--input',str(out/'input/patient.csv'),
                    '--models-dir',str(out/'models'),'--output-dir',str(out/'example_prediction'),'--plots'],check=True,stdout=subprocess.DEVNULL)
    sample={f'Q{i+1}':None if __import__('pandas').isna(df.iloc[0][f]) else float(df.iloc[0][f]) for i,f in enumerate(FEATURES)}
    write_json({'name':'Synthetic demo','referenceKey':'demo001','answers':sample},out/'input/patient_request.json')
    write_json({'synthetic':True,'completed_tasks':len(metrics),'metrics':metrics,'elapsed_seconds':round(time.monotonic()-started,2),
                'note':'Software demonstration only. These are not the study models or published results.'},out/'demo_summary.json')
    print(f'Completed all {len(metrics)} synthetic tasks. Outputs: {out}')

if __name__=='__main__': main()
