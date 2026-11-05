# MALCO heart-failure trial research code

**MALCO** (Multi-Agent LLM-powered Collaborative Outpatient Care) accompanies the manuscript *Multi-agent Large Language Model Framework Integrating Prognostic Machine Learning for Personalized Heart Failure Management: a Multi-center Randomized Controlled Trial*.

This repository provides a runnable research demonstration: eight XGBoost prediction tasks, numerical SHAP explanations, a three-agent API workflow, and R analyses. **The default demonstration uses synthetic data and newly trained demonstration models.** Offline conversation and review responses are fixed, labelled test responses. Patient-level clinical data and the prognostic model weights used in the study are not included.

[Running instructions](QUICKSTART.md) · [Data format](data/DATA_FORMAT.md) · [Validation record](VALIDATION.txt) · [Changes and checks](REPAIR_CHECKLIST.md)

## What you can run

| Component | Input | Output |
|---|---|---|
| Prognostic modelling | 41 predictors and a selected binary endpoint | Saved model, metadata, validation predictions and ROC plot |
| Patient Portfolio Agent | One patient's 41 predictor fields | Eight predictions and numerical SHAP contributions |
| Conversational Agent | Predictions, SHAP and conversation history | Streamed responses; offline example or configured live provider |
| Review Panel Agent | Completed conversation | Three reviewer verdicts and an aggregate decision |
| Statistical analyses | Documented prediction, questionnaire or rating tables | ROC/KM, AUC comparisons, questionnaire and ICC outputs, ablation tests and plots |

The landing page is informational. The complete demonstration is exercised through the documented API requests. This repository does not provide a patient-facing clinical service.

## Quick start

Install Git, Python and Node.js before starting. R is optional for the statistical examples.

| Runtime | Supported/requested version | Locally tested version |
|---|---|---|
| Python | 3.11–3.13; use the supplied dependency lock | 3.13.5 |
| Node.js | >=20.9 | 24.13.0 |
| R (optional) | >=4.3 | 4.5.2 with the existing package library |

Windows PowerShell (tested on Windows 11):

```powershell
git clone https://github.com/MeredithcxWang/MALCO_RCT_Study.git
cd MALCO_RCT_Study
.\setup.bat
.\run_ml_pipeline.bat
.\start_malco_demo.bat
```

Linux/macOS (launch scripts supplied; execution on these platforms has not been verified):

```bash
git clone https://github.com/MeredithcxWang/MALCO_RCT_Study.git
cd MALCO_RCT_Study
bash setup.sh
bash run_ml_pipeline.sh
bash start_malco_demo.sh
```

`setup` creates an isolated `.venv` and installs locked Python and Node dependencies. R is optional and installed separately. `run_ml_pipeline` generates independent synthetic training/evaluation samples, trains all eight demonstration tasks, reloads the models, evaluates them, and produces single-patient predictions with numerical SHAP contributions and plots. `start_malco_demo` binds the API to localhost:3000 in offline demo mode.

No API key or GPU is required for the default offline demonstration. Internet access is needed for dependency installation. Keep the service terminal open and visit [the local landing page](http://127.0.0.1:3000). In a second PowerShell terminal, check the service with:

```powershell
Invoke-RestMethod http://127.0.0.1:3000/api/health
```

Expected: `status: ok`, `mode: demo`, and eight tasks. A successful model run writes `output/demo/demo_summary.json` with `completed_tasks: 8`, eight model/metadata pairs, internal and synthetic evaluation predictions, and the single-patient SHAP example. Use [QUICKSTART.md](QUICKSTART.md) to send the sample patient, chat and review requests.

## Inputs and artifacts

- `data/schema.json`: authoritative ordered list of **41 predictors**, eight task names, and the `rehospitalization` → `rehos` alias.
- `data/synthetic_sample.xlsx`: original public example (100 rows, 41 predictors plus two outcome columns). It is preserved unchanged.
- `generate_synthetic_data.py`: creates clearly labelled synthetic examples under `output/demo/input/`, including the inputs for all R analyses. It does not overwrite the original public example.
- `ml_prognostic_models/xgboost_training.py`: selected-task training, stratified holdout split, deterministic search, saved model/metadata and internal predictions.
- `ml_prognostic_models/model_evaluation.py`: evaluates one explicitly selected model on supplied data. Missing inputs fail with a nonzero exit code.
- `ml_prognostic_models/ensemble_imputation.py`: standalone imputation example; targets are excluded and observed values retained. The runnable model demonstration instead uses XGBoost's native missing-value support, consistently during training and inference. Do not fit this standalone imputer on the combined training and test cohorts.
- `malco_framework/`: complete minimal Next.js API project, three agents, configuration and Python inference.
- `statistical_analysis/`: R scripts with command-line paths and synthetic examples.

The ablation script supports the original two-condition example (five comparisons) and `--design four-stage` (all six stage pairs across five metrics, 30 comparisons). Both modes export full-precision raw/adjusted P values and use only adjusted P values for figure stars. Real ratings are supplied separately in the documented format.

For both ablation modes, the two raters must score the same patients, stages and metrics. Pairing uses patient IDs; Bonferroni correction is applied to the complete comparison family before adjacent-stage results are selected for labels. See [the rating format](data/DATA_FORMAT.md#four-stage-ablation-ratings).

All new model artifacts use `models/<task>/model.json` plus `metadata.json`, which records predictor order, preprocessing, threshold, seed, input hash, search settings and package versions. The new implementation uses **no scaler**, consistently at both training and inference. Do not attach an arbitrary scaler to these models. Historical pickle/scaler artifacts require a separately validated conversion; they are not automatically treated as this new format.

Python and R share `<task>_prediction_results.xlsx`, `True_Label`, `Predicted_Probability` and `Predicted_Class`. The `internal/` and `external/` folder names are software conventions; synthetic evaluation is not the manuscript's external clinical validation.

## Custom training

In the commands below, `python` means the repository's virtual-environment interpreter: `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux/macOS.

```bash
python ml_prognostic_models/xgboost_training.py --input /path/to/train.xlsx --task 2yrM --output-dir output/custom --cv 10 --search-iterations 50
python ml_prognostic_models/model_evaluation.py --input /path/to/evaluation.xlsx --task 2yrM --models-dir output/custom/models --output-dir output/custom
python malco_framework/run_models.py --input /path/to/one_patient.csv --models-dir output/custom/models --tasks 2yrM --output-dir output/custom/prediction --plots
```

All 41 feature columns are required; individual missing values are allowed. Outcomes and all `Days_to_*` fields are excluded through a feature allowlist. CSV and XLSX are accepted. Paths are resolved against the repository, independent of the current working directory.

Demo training uses two candidates and three-fold CV for fast software verification. Non-demo training retains the original published script's search distributions and defaults to 50 candidates and ten folds. The holdout fraction is configurable with `--test-size` (default 0.2). These settings and the default 0.5 risk threshold are not claims about the historical trial model. Validate the actual research configuration separately.

## Offline and live language-model modes

The default `MALCO_MODE=demo` makes no LLM requests. It uses real synthetic-model inference and fixed, clearly labelled dialogue/review test responses. To run live, copy `.env.example` to `malco_framework/.env.local`, set `MALCO_MODE=live`, provider endpoints, credentials and supported model IDs. See QUICKSTART for variable names. Credentials are never included in the upload package.

The live review panel retains the original A/B selection probability (40%/60%) and historical model identifiers as configuration defaults. Providers may not support those identifiers now. Override them using `REVIEW_CONFIGURATIONS` (A/B) or `REVIEW_MODELS` (one explicit three-reviewer set). A current demonstration is not evidence that a historical model snapshot is still available. Incomplete, invalid or failed reviews are recorded and cannot produce a passing verdict.

## Statistical examples

After generating the demonstration inputs, install the optional R environment with `Rscript statistical_analysis/setup.R`, or use an existing compatible R library. The package snapshot and restoration instructions are included; a complete fresh R restoration has not yet been verified.

```bash
Rscript statistical_analysis/DeLong_Test_analysis.R
Rscript statistical_analysis/ROC_with_survival_analysis.R
Rscript statistical_analysis/questionnaire_results_analysis.R
Rscript statistical_analysis/review_grade_ICC_analysis.R
Rscript statistical_analysis/ablation_study_analysis.R
Rscript statistical_analysis/ablation_study_analysis.R --design four-stage
```

Outputs are written under `output/demo/statistics/`. The four-stage ablation exports all 30 tests, 15 adjacent comparisons, mean scores and a figure. The generated rating examples are synthetic; reproducing study values requires the corresponding research data.

## Verification and reproducibility scope

After setup and generation of the demo artifacts, run these checks from the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_pipeline.py -v
node --test tests/api.test.mjs
npm --prefix malco_framework run build
```

With the optional R environment, also run `Rscript tests/statistics.R`. On Linux/macOS, substitute `.venv/bin/python` for the Windows Python path.

API tests include a localhost test service for live-adapter failure, veto and truncation checks; they do not contact a real provider. The [GitHub Actions workflow](.github/workflows/demo.yml) runs the Python/Node demonstration and checks on Windows; R checks are separate. Check [workflow runs](https://github.com/MeredithcxWang/MALCO_RCT_Study/actions/workflows/demo.yml) for the result of a particular commit.

The [validation record](VALIDATION.txt) lists tests performed and remaining limits. Passing synthetic workflows establishes software execution, not reproduction of clinical performance. Historical preprocessing, model weights, clinical predictions and original LLM snapshots require separate provenance and compatibility checks. Real-provider availability, a fresh R restoration and non-Windows execution remain unverified.

When reporting or reproducing an analysis, record the code revision (`git rev-parse HEAD`), commands, input provenance and dependency versions. Do not substitute generated demonstration results for manuscript results.
