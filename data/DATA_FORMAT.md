# Data format

## Model inputs and outputs

- CSV/XLSX inputs must include the 41 numeric predictors listed in [schema.json](schema.json). Training/evaluation also require the selected binary outcome. Outcomes and `Days_to_*` columns are excluded from predictors.
- Tasks: `2yrM`, `3yrM`, `4yrM`, `5yrM`, `Cancer`, `COPD`, `Pneumonia`, `rehos`. The CLI alias `rehospitalization` maps to `rehos`.
- Individual missing values are allowed; missing columns, entirely empty patient rows and nonnumeric/infinite values are rejected. Units are not converted automatically.
- API answers `Q1`–`Q41` follow schema order; `A`/`B` mean 1/0. A `referenceKey` contains 1–128 letters, digits, underscores or hyphens.
- Predictions: `<task>_prediction_results.xlsx`, with `True_Label`, `Predicted_Probability`, `Predicted_Class`. Saved models use `model.json` plus matching `metadata.json`; historical pickle/scaler files need separate conversion.

The public sample contains 100 synthetic rows: 41 predictors, `2yrM` and `Days_to_2yrM`. Generated inputs, ratings and follow-up times under `output/demo/input/` are also synthetic.

## Predictor coding and units

`use.Sex`: 0=female, 1=male; `use.age`: years. Other binary predictors use 0=no, 1=yes (obesity: BMI ≥30; medication: prescribed). Units below are retained from the example documentation and must be checked against the research data source.

| Predictor | Unit |
|---|---|
| `Hemoglobin.base.value` | g/dL |
| `HbA1c.base.value` | % |
| `Glucose.base.value`, `use.TG.base.result`, `LDL.1y.Numeric.Result` | mmol/L |
| `Creatinie.base.value`, `TIBC.base.result`, `Iron.base.Result` | μmol/L |
| `Lymph.base.value`, `Neurophil.base.value` | 10⁹/L |
| `calcu.TSAT.all`, `Iron.saturation.value` | % |
| `FerritinugL` | μg/L |
| `CRP.value` | mg/L |
| `mean.BMI` | kg/m² |
| `use.ALT.base.result`, `AST` | U/L |
| `eGFR` | mL/min/1.73m² |
| `Albumin_Result` | Source unit requires verification; original synthetic range 2.5–5 |

## Four-stage ablation ratings

Supply a CSV with `PatientID,Rater,Model,Metric,Score`: one integer score (1–5) per row. Exactly two raters must score every patient/model/metric combination once; duplicate or incomplete entries are rejected. Pairing uses patient IDs, not row order.

- Models: `V1` baseline, `V2` + safety prompt, `V3` + ML raw data, `V4` + SHAP.
- Metrics: `Accuracy`, `Safety`, `Completeness`, `Personalization`, `Adherence` (guideline adherence, not medication PDC).
- Ratings are averaged within patient/model/metric. Two-sided paired Wilcoxon tests use a normal approximation and continuity correction; identical pairs give P=1. Bonferroni correction covers all 30 tests before selecting the 15 adjacent comparisons for labels.

Use `--design four-stage --input /path/to/ratings.csv --output-dir /path/to/results`. Outputs include `paired_tests.csv`, `adjacent_tests.csv`, `mean_scores.csv` and the plot. The default two-condition mode reads the generated `Review_doc_GST.xlsx` and `Review_doc_CGH.xlsx` files from `output/demo/input/ablation/`.
