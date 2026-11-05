"""Generate deterministic, purely synthetic demonstration inputs; never clinical data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from malco_core import FEATURES, TASKS, resolve_path, write_table, write_json
RANGES = {'use.age': (50.0, 95.0), 'Hemoglobin.base.value': (8.0, 16.0), 'HbA1c.base.value': (4.5, 12.0), 'Glucose.base.value': (4.0, 15.0), 'Creatinie.base.value': (50.0, 250.0), 'Lymph.base.value': (0.5, 4.0), 'Neurophil.base.value': (2.0, 15.0), 'calcu.TSAT.all': (10.0, 50.0), 'TIBC.base.result': (20.0, 80.0), 'Iron.base.Result': (3.0, 30.0), 'FerritinugL': (50.0, 2000.0), 'CRP.value': (1.0, 50.0), 'Iron.saturation.value': (10.0, 50.0), 'use.TG.base.result': (0.5, 5.0), 'mean.BMI': (18.0, 35.0), 'use.ALT.base.result': (5.0, 80.0), 'LDL.1y.Numeric.Result': (1.5, 5.0), 'Albumin_Result': (2.5, 5.0), 'AST': (15.0, 100.0), 'eGFR': (15.0, 120.0)}

def generate(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.uniform(*RANGES[f], n) if f in RANGES else rng.integers(0, 2, n) for f in FEATURES})
    score = (df['use.age'] - 72) / 20 + .4 * df['use.DM'] - .2 * (df['eGFR'] - 60) / 40
    for i, task in enumerate(TASKS):
        chance = 1 / (1 + np.exp(-(score + (i - 4) * .08)))
        event = rng.binomial(1, chance)
        horizon = int(task[0]) * 365 if 'yrM' in task else 730
        df[task] = event
        df['Days_to_' + task] = np.where(event == 1, rng.integers(1, horizon, n), horizon)
    df['Death.24'] = df['5yrM']  # Synthetic 5-year mortality only, for the legacy example plot.
    for col in ('Creatinie.base.value', 'Glucose.base.value', 'Iron.base.Result'):
        df.loc[rng.random(n) < .1, col] = np.nan
    return df

def statistical_inputs(directory, seed):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, 31)
    survey = pd.DataFrame({'PatientID':np.arange(1,61), 'Group_Match':['pre']*30+['post']*30,
                          'S24':rng.integers(1,6,60),'S25':rng.integers(1,6,60)})
    write_table(survey, directory/'survey_data.xlsx')
    metrics = ['Accuracy Score (Likert 1~5)','Safety (Likert 1~5)','Relevance (Likert 1~5)',
               'Empathy (Likert 1~5)','Completeness (Likert 1~5)','Usefulness (Likert 1~5)','Adherence to Guidelines (Likert 1~5)']
    latent = rng.integers(2,5,(30,7))
    for rater in range(1,5):
        values = np.clip(latent+rng.integers(-1,2,(30,7)),1,5)
        frame = pd.DataFrame(values,columns=metrics)
        frame.insert(0,'Rater',f'R{rater}')
        frame.insert(0,'Patient ID',ids)
        write_table(frame,directory/'grades'/f'FINAL_R{rater}.xlsx')
    for rater in ('GST','CGH'):
        # Matches the published two-condition script's positional format, including one title row.
        frame = pd.DataFrame({'PatientID':ids,'Description':['SYNTHETIC']*30})
        for metric in ('Accuracy','Safety','Completeness','Personalization','Adherence'):
            frame['My_'+metric] = rng.integers(2,6,30)
        frame['Separator'] = ''
        for metric in ('Accuracy','Safety','Completeness','Personalization','Adherence'):
            frame['Base_'+metric] = rng.integers(1,6,30)
        path=directory/'ablation'/f'Review_doc_{rater}.xlsx'
        path.parent.mkdir(parents=True,exist_ok=True)
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame([['SYNTHETIC TEST INPUT - NOT PHYSICIAN RATINGS']]).to_excel(writer,index=False,header=False)
            frame.to_excel(writer,index=False,startrow=1)
    # Separate four-stage software fixture, unrelated to study rating values.
    rows = []
    for patient in ids:
        for rater in ('R1', 'R2'):
            for model in ('V1', 'V2', 'V3', 'V4'):
                for metric in ('Accuracy','Safety','Completeness','Personalization','Adherence'):
                    rows.append({'PatientID':f'SYNTHETIC-{patient}', 'Rater':rater,
                                 'Model':model, 'Metric':metric, 'Score':int(rng.integers(1,6))})
    write_table(pd.DataFrame(rows), directory/'ablation_four_stage.csv')
    write_json({'synthetic':True,'purpose':'Software tests only; not study results','seed':seed},directory/'SYNTHETIC.json')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='output/demo/input')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-rows',type=int,default=240)
    p.add_argument('--evaluation-rows',type=int,default=120)
    args=p.parse_args()
    if min(args.train_rows,args.evaluation_rows)<30: p.error('At least 30 rows per dataset are required')
    dest=resolve_path(args.output_dir)
    write_table(generate(args.train_rows,args.seed),dest/'train.xlsx')
    write_table(generate(args.evaluation_rows,args.seed+1),dest/'evaluation.xlsx')
    statistical_inputs(dest,args.seed+2)
    print(f'Created synthetic inputs in {dest}; not evidence for clinical performance.')

if __name__=='__main__': main()
