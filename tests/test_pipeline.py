import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from malco_core import FEATURES,TASKS,feature_frame,read_table,load_artifact
from ml_prognostic_models.ensemble_imputation import ensemble_imputation_custom

class PipelineTests(unittest.TestCase):
    def test_task_alias_loads_canonical_artifact(self):
        model, meta = load_artifact(ROOT/'output/demo/models', 'rehospitalization')
        self.assertEqual(meta['task'], 'rehos')
    def test_no_outcome_features_and_column_reordering(self):
        df=read_table(ROOT/'output/demo/input/train.xlsx')
        expected=feature_frame(df)
        self.assertEqual(list(expected),FEATURES)
        self.assertEqual(expected.shape[1],41)
        pd.testing.assert_frame_equal(expected,feature_frame(df.iloc[:,::-1]))
        self.assertFalse(any(c.startswith('Days_to_') or c in TASKS for c in expected))
    def test_missing_column_and_empty_patient_fail(self):
        df=read_table(ROOT/'output/demo/input/train.xlsx')
        with self.assertRaises(ValueError):feature_frame(df.drop(columns=[FEATURES[0]]))
        with self.assertRaises(ValueError):feature_frame(pd.DataFrame([[None]*41],columns=FEATURES))
    def test_complete_data_imputation_is_noop(self):
        df=read_table(ROOT/'data/synthetic_sample.xlsx').loc[:,FEATURES].fillna(0)
        pd.testing.assert_frame_equal(df,ensemble_imputation_custom(df,[]))
    def test_all_artifacts_and_shap_additivity(self):
        result=json.loads((ROOT/'output/demo/example_prediction/predictions.json').read_text())
        self.assertEqual(len(result['predictions']),8)
        for p in result['predictions']:
            model,meta=load_artifact(ROOT/'output/demo/models',p['task'])
            self.assertEqual(meta['features'],FEATURES)
            self.assertTrue(meta['synthetic_demo'])
            explanation=p['shap']
            margin=explanation['base_value']+sum(v['shap_value'] for v in explanation['contributions'])
            self.assertAlmostEqual(1/(1+math.exp(-margin)),p['value'],places=5)
            self.assertGreater((ROOT/f"output/demo/external/{p['task']}/{p['task']}_prediction_results.xlsx").stat().st_size,0)
    def test_imputation_preserves_observed_values_and_outcomes(self):
        source=read_table(ROOT/'data/synthetic_sample.xlsx')
        path=ROOT/'output/demo/imputed_sample.xlsx'
        if not path.exists():self.skipTest('Run the standalone imputation example first')
        result=read_table(path)
        self.assertFalse(result[FEATURES].isna().any().any())
        for col in source:
            mask=source[col].notna()
            pd.testing.assert_series_equal(source.loc[mask,col],result.loc[mask,col],check_dtype=False)
    def test_missing_input_exits_nonzero(self):
        r=subprocess.run([sys.executable,str(ROOT/'ml_prognostic_models/model_evaluation.py'),'--input','does-not-exist.csv'],capture_output=True)
        self.assertNotEqual(r.returncode,0)
    def test_artifact_mismatch_fails(self):
        with self.assertRaises(FileNotFoundError):load_artifact(ROOT/'output/no-such-models','2yrM')

if __name__=='__main__':unittest.main()
