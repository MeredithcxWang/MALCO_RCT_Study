"""Single-patient inference with structured native TreeSHAP contributions.
stdout contains exactly one JSON response; diagnostic errors go to stderr.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import xgboost as xgb
from malco_core import FEATURES, TASKS, resolve_path, task_name, read_table, feature_frame, load_artifact, write_json

def infer(input_path, models_dir, tasks, output_dir=None, plots=False):
    X=feature_frame(read_table(input_path))
    if len(X)!=1: raise ValueError('Inference requires exactly one patient row')
    loaded=[(task_name(task),*load_artifact(models_dir,task_name(task))) for task in tasks]
    predictions=[]
    for task,model,meta in loaded:
        probability=float(model.predict_proba(X)[0,1])
        if not np.isfinite(probability) or not 0<=probability<=1: raise ValueError('Invalid prediction')
        contributions=model.get_booster().predict(xgb.DMatrix(X),pred_contribs=True)[0]
        base=float(contributions[-1])
        values=contributions[:-1]
        threshold=meta['threshold']
        risk=('Low' if probability<=.5*threshold else 'Low-Medium' if probability<=.8*threshold else
              'Medium-High' if probability<=1.2*threshold else 'High')
        explanation={'units':'log_odds','base_value':base,'contributions':[
            {'feature':f,'value':None if np.isnan(float(X.iloc[0][f])) else float(X.iloc[0][f]),'shap_value':float(v)}
            for f,v in zip(FEATURES,values)],'interpretation':'Model associations, not causal effects.'}
        predictions.append({'task':task,'model':TASKS[task],'value':probability,'risk':risk,
                            'synthetic_demo':meta['synthetic_demo'],'shap':explanation})
        if plots:
            if output_dir is None: raise ValueError('--plots requires --output-dir')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import shap
            output_dir.mkdir(parents=True,exist_ok=True)
            shap.force_plot(base,values,X.iloc[0],matplotlib=True,show=False)
            fig=plt.gcf()
            fig.set_size_inches(20,3)
            fig.savefig(output_dir/f'{task}_force.png',bbox_inches='tight',dpi=120)
            plt.close(fig)
    result={'schema_version':1,'predictions':predictions}
    if output_dir is not None: write_json(result,output_dir/'predictions.json')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--models-dir',default='output/demo/models')
    p.add_argument('--tasks',default=','.join(TASKS))
    p.add_argument('--output-dir')
    p.add_argument('--plots',action='store_true')
    args=p.parse_args()
    try:
        tasks=[t.strip() for t in args.tasks.split(',') if t.strip()]
        if not tasks: raise ValueError('At least one task must be requested')
        result=infer(resolve_path(args.input),resolve_path(args.models_dir),tasks,
                     resolve_path(args.output_dir) if args.output_dir else None,args.plots)
        print(json.dumps(result,allow_nan=False))
    except Exception as exc: p.exit(1,f'Inference failed: {exc}\n')

if __name__=='__main__': main()
