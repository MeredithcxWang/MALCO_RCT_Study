# Data contracts

`schema.json` is the authoritative ordered list of **41 clinical predictor columns**. Training and evaluation also require the selected binary task column. Event-time fields are optional for prediction, but required for time-to-event figures. Neither outcome nor time-to-event columns can enter the predictor matrix.

The original `synthetic_sample.xlsx` has 100 rows and 43 columns: 41 predictors, `2yrM`, and `Days_to_2yrM`. These are not 43 predictors. It remains unchanged.

Task names: `2yrM`, `3yrM`, `4yrM`, `5yrM`, `Cancer`, `COPD`, `Pneumonia`, `rehos`. `rehospitalization` is accepted as a command-line alias for `rehos`; data columns use `rehos` and `Days_to_rehos`.

All 41 feature columns must be present, numeric, and in the documented units of the source dataset. Individual cells may be missing (blank/NaN). Entirely empty patient rows and nonnumeric/infinite values are rejected. No automatic unit conversion is applied; source units, including albumin, must be verified for research use.

API answers map `Q1`–`Q41` to `schema.json` in order. Missing answers become blank cells. `A` and `B` are retained as aliases for 1 and 0; numerical JSON values are preferred. Supply `referenceKey` using 1–128 letters, digits, underscores or hyphens.

All generated files under `output/demo/input/` are synthetic, including the apparent physician ratings. Their IDs identify synthetic rows, not patients. Training and evaluation samples use separate random seeds. Event/censoring times are artificial and cannot support clinical inference.

Prediction files are named `<task>_prediction_results.xlsx` and use `True_Label`, `Predicted_Probability` and `Predicted_Class`. Internal and supplied-evaluation files use the same schema. R analysis-specific inputs and generation details are in `generate_synthetic_data.py`; see QUICKSTART for paths.

The model metadata states the preprocessing, ordered predictors, threshold and synthetic status. Older research pickle/scaler pairs are not silently converted into the new format.

## Four-stage ablation ratings

Use a CSV with columns `PatientID,Rater,Model,Metric,Score`, one integer score (1-5) per row. PatientID is a stable deidentified string. There must be exactly two raters and one rating from each for every patient, model and metric. Duplicate keys, missing ratings and differing patient sets are errors; row order does not define pairing.

Models: `V1` (Gemini baseline), `V2` (+ safety prompt), `V3` (+ ML raw data), `V4` (+ SHAP). Metrics: `Accuracy`, `Safety`, `Completeness`, `Personalization`, `Adherence` (guideline-adherence rating, not trial medication PDC). The script averages raters within patient, calculates all 30 pairwise tests, applies one Bonferroni family, then selects adjacent comparisons for labels. Actual study data require independent provenance verification; changing column labels does not establish provenance.

## Predictor descriptions retained from the source

The ranges below describe the supplied synthetic example, not validated clinical acceptance limits. Verify source units before any research use.

### 1. Demographics (2 features)
- `use.Sex`: Binary (0=Female, 1=Male)
- `use.age`: Continuous (range: 50-95 years)

### 2. Lifestyle & Comorbidities (13 features)
All binary (0=No, 1=Yes):
- `use.Alcohol`: Alcohol consumption
- `use.Smoke`: Smoking status
- `use.HT`: Hypertension
- `use.DM`: Diabetes mellitus
- `use.obesity`: Obesity (BMI ≥ 30)
- `use.CAD`: Coronary artery disease
- `use.PVD`: Peripheral vascular disease
- `use.AF`: Atrial fibrillation
- `use.DL`: Dyslipidemia
- `use.Anemia`: Anemia
- `use.cirrhosis`: Liver cirrhosis
- `use.parkinson`: Parkinson's disease
- `use.rheumatism`: Rheumatic disease

### 3. Pharmacotherapy (7 features)
All binary (0=Not prescribed, 1=Prescribed):
- `use.ACEI.base`: ACE inhibitors
- `use.ARB.base`: Angiotensin receptor blockers
- `use.Bblocker.base`: Beta-blockers
- `use.ASA.base`: Aspirin
- `use.CCB.base`: Calcium channel blockers
- `use.duretics.base`: Diuretics
- `statin_base`: Statins

### 4. Laboratory Parameters (19 features)
All continuous, may contain missing values (NaN):
- `Hemoglobin.base.value`: g/dL (range: 8-16)
- `HbA1c.base.value`: % (range: 4.5-12)
- `Glucose.base.value`: mmol/L (range: 4-15)
- `Creatinie.base.value`: μmol/L (range: 50-250)
- `Lymph.base.value`: 10⁹/L (range: 0.5-4)
- `Neurophil.base.value`: 10⁹/L (range: 2-15)
- `calcu.TSAT.all`: Transferrin saturation % (range: 10-50)
- `TIBC.base.result`: μmol/L (range: 20-80)
- `Iron.base.Result`: μmol/L (range: 3-30)
- `FerritinugL`: μg/L (range: 50-2000)
- `CRP.value`: mg/L (range: 1-50)
- `Iron.saturation.value`: % (range: 10-50)
- `use.TG.base.result`: Triglycerides, mmol/L (range: 0.5-5)
- `mean.BMI`: kg/m² (range: 18-35)
- `use.ALT.base.result`: U/L (range: 5-80)
- `LDL.1y.Numeric.Result`: mmol/L (range: 1.5-5)
- `Albumin_Result`: original synthetic numeric range 2.5-5; source unit requires verification. No conversion is applied.
- `AST`: U/L (range: 15-100)
- `eGFR`: mL/min/1.73m² (range: 15-120)
