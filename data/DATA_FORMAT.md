# Data Format Specification

## Overview

This document describes the input data format required by the MALCO framework's machine learning pipeline and inference engine. All patient data files must follow this schema for compatibility with the XGBoost models and three-agent system.

---

## File Format Requirements

- **Supported formats**: Excel (`.xlsx`) or CSV (`.csv`)
- **Encoding**: UTF-8
- **Row structure**: One patient per row
- **Column structure**: 41 clinical features + 2 outcome variables (for training data)

**Sample file provided**: `synthetic_sample.xlsx` contains 100 synthetically generated patient profiles demonstrating the exact format expected by the pipeline.

---

## Feature Schema (41 Clinical Features)

### 1. Demographics (2 features)

| Feature Name | Type | Description | Range/Values |
|-------------|------|-------------|--------------|
| `use.Sex` | Binary | Sex | 0 = Female, 1 = Male |
| `use.age` | Continuous | Age in years | 50–95 |

---

### 2. Lifestyle & Comorbidities (13 features)

All binary features (0 = No/Absent, 1 = Yes/Present):

| Feature Name | Description |
|-------------|-------------|
| `use.Alcohol` | Alcohol consumption |
| `use.Smoke` | Smoking status |
| `use.HT` | Hypertension |
| `use.DM` | Diabetes mellitus |
| `use.obesity` | Obesity (BMI ≥ 30 kg/m²) |
| `use.CAD` | Coronary artery disease |
| `use.PVD` | Peripheral vascular disease |
| `use.AF` | Atrial fibrillation |
| `use.DL` | Dyslipidemia |
| `use.Anemia` | Anemia |
| `use.cirrhosis` | Liver cirrhosis |
| `use.parkinson` | Parkinson's disease |
| `use.rheumatism` | Rheumatic disease |

---

### 3. Pharmacotherapy (7 features)

All binary features (0 = Not prescribed, 1 = Prescribed):

| Feature Name | Description |
|-------------|-------------|
| `use.ACEI.base` | ACE inhibitors at baseline |
| `use.ARB.base` | Angiotensin receptor blockers at baseline |
| `use.Bblocker.base` | Beta-blockers at baseline |
| `use.ASA.base` | Aspirin at baseline |
| `use.CCB.base` | Calcium channel blockers at baseline |
| `use.duretics.base` | Diuretics at baseline |
| `statin_base` | Statins at baseline |

---

### 4. Laboratory Parameters (19 features)

All continuous features, may contain missing values (represented as `NaN` in Excel or empty cells in CSV):

| Feature Name | Description | Unit | Typical Range |
|-------------|-------------|------|--------------|
| `Hemoglobin.base.value` | Hemoglobin | g/dL | 8–16 |
| `HbA1c.base.value` | Glycated hemoglobin | % | 4.5–12 |
| `Glucose.base.value` | Blood glucose | mmol/L | 4–15 |
| `Creatinie.base.value` | Creatinine | μmol/L | 50–250 |
| `Lymph.base.value` | Lymphocyte count | 10⁹/L | 0.5–4 |
| `Neurophil.base.value` | Neutrophil count | 10⁹/L | 2–15 |
| `calcu.TSAT.all` | Transferrin saturation (calculated) | % | 10–50 |
| `TIBC.base.result` | Total iron-binding capacity | μmol/L | 20–80 |
| `Iron.base.Result` | Serum iron | μmol/L | 3–30 |
| `FerritinugL` | Ferritin | μg/L | 50–2000 |
| `CRP.value` | C-reactive protein | mg/L | 1–50 |
| `Iron.saturation.value` | Iron saturation | % | 10–50 |
| `use.TG.base.result` | Triglycerides | mmol/L | 0.5–5 |
| `mean.BMI` | Body mass index | kg/m² | 18–35 |
| `use.ALT.base.result` | Alanine aminotransferase | U/L | 5–80 |
| `LDL.1y.Numeric.Result` | LDL cholesterol | mmol/L | 1.5–5 |
| `Albumin_Result` | Serum albumin | g/L | 2.5–5 |
| `AST` | Aspartate aminotransferase | U/L | 15–100 |
| `eGFR` | Estimated glomerular filtration rate | mL/min/1.73m² | 15–120 |

---

### 5. Outcome Variables (for training data only)

For **model training and validation**, include outcome columns corresponding to the prediction task:

**Example: 2-year mortality prediction**
- `Days_to_2yrM`: Time to event or censoring (days, range: 30–730)
- `2yrM`: 2-year mortality outcome (0 = Alive/Censored, 1 = Deceased)

**Other prediction tasks:** Replace with corresponding outcome variables:
- 3-year mortality: `Days_to_3yrM`, `3yrM`
- 4-year mortality: `Days_to_4yrM`, `4yrM`
- 5-year mortality: `Days_to_5yrM`, `5yrM`
- Cancer: `Days_to_Cancer`, `Cancer`
- COPD: `Days_to_COPD`, `COPD`
- Pneumonia: `Days_to_Pneumonia`, `Pneumonia`
- Rehospitalization: `Days_to_rehospitalization`, `rehospitalization`


---

## Missing Data Handling

- **Representation**: Missing values should be encoded as `NaN` (Excel) or left as empty cells (CSV)
- **Imputation**: The ML pipeline uses ensemble imputation (KNN + IterativeImputer + PCA) before model training
- **Implementation**: See `ml_prognostic_models/ensemble_imputation.py`

**Important**: XGBoost models in the MALCO framework handle missing values natively during inference via the `missing=np.nan` parameter, so raw patient data can be submitted to the Patient Portfolio Agent without pre-imputation.

---

## Column Naming Conventions

Understanding the naming patterns helps when preparing new datasets:

- **`.base` suffix**: Baseline measurement at hospital admission (e.g., `Hemoglobin.base.value`)
- **`.value` / `.result` / `.Result`**: Numeric laboratory results
- **`use.` prefix**: Clinical usage indicators (medications, conditions)
- **`calcu.` prefix**: Calculated or derived features (e.g., `calcu.TSAT.all`)

**Critical**: Column names are **case-sensitive** and must match exactly. The XGBoost models were trained with these exact feature names.

---

## Data Privacy & Ethical Constraints

### Synthetic Sample Data

The provided file **`synthetic_sample.xlsx`** contains **100 synthetically generated patient profiles** created for demonstration and testing purposes. These profiles:
- Were generated using statistical distributions derived from the real cohort
- Do **NOT** represent actual patients
- Are provided to enable code testing and methodological review

### Real Patient Data (Not Included)

The actual clinical data used in the published trial results:
- Contains **protected health information (PHI)** under HIPAA and local privacy regulations
- Is governed by **IRB Protocol** and institutional data use agreements
- **Cannot be publicly shared** due to ethical and legal constraints
- May be accessible through **controlled data access** for qualified researchers who:
  - Have IRB approval from their institution
  - Sign a data use agreement with the data custodian
  - Provide a scientifically justified analysis plan

---

## Validation Checklist

Before running the ML pipeline or MALCO framework on new data, verify:

- [ ] **All 41 feature columns are present** with exact names (case-sensitive)
- [ ] **Binary features** contain only `0`, `1`, or `NaN` (no other values)
- [ ] **Continuous features** are numeric and within plausible clinical ranges
- [ ] **Outcome variables** (if included for training) are properly coded (0/1)
- [ ] **No duplicate patient records** (check for duplicate IDs if present)
- [ ] **File encoding is UTF-8** (avoid Excel's legacy encodings)
- [ ] **No formula injection vulnerabilities** (CSV cells should not start with `=`, `+`, `-`, `@`)

---

## Integration with MALCO Framework

### For Training (ML Pipeline)
1. Place training data files in `ml_prognostic_models/data/`
2. Ensure outcome columns are included
3. Run `ensemble_imputation.py` → `xgboost_training.py` → `model_evaluation.py`

### For Inference (Patient Portfolio Agent)
1. Patient data submitted via questionnaire (41 features, no outcome columns)
2. Patient Portfolio Agent maps questionnaire responses to feature names
3. Single-row CSV generated and passed to `run_models.py`
4. Python subprocess loads trained models and generates predictions + SHAP plots
5. Results returned to frontend via JSON

**Reference**: See `malco_framework/patient_portfolio_agent.js` for the `FEATURE_MAP` object that maps questionnaire items (Q1–Q41) to these feature names.

---


## Additional Resources

- **Feature engineering details**: See manuscript Methods section, Table 1
- **Imputation algorithm**: `ml_prognostic_models/ensemble_imputation.py`
- **Model training code**: `ml_prognostic_models/xgboost_training.py`
- **Questionnaire mapping**: `malco_framework/patient_portfolio_agent.js` (`FEATURE_MAP` object)
- **SHAP explainability**: `malco_framework/run_models.py` (SHAP force plot generation)

---
