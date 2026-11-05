# Reproduce the public demonstration

Use the environment installed by `setup.bat` or `bash setup.sh`. On Windows the Python executable is `.venv/Scripts/python.exe`; on Linux/macOS it is `.venv/bin/python`.

## 1. Generate models and predictions

Run `run_ml_pipeline.bat` (Windows) or `bash run_ml_pipeline.sh`. Alternatively:

```bash
python scripts/run_demo.py
```

Expected: eight model/metadata pairs in `output/demo/models/`, eight internal and eight synthetic evaluation prediction workbooks, eight ROC images, numerical SHAP and force plots in `output/demo/example_prediction/`, and `output/demo/demo_summary.json` with `completed_tasks: 8`.

The generated patient, questionnaire and physician-rating examples are entirely synthetic software fixtures. Do not use them in manuscript result tables.

Optional standalone imputation check:

```bash
python ml_prognostic_models/ensemble_imputation.py --input data/synthetic_sample.xlsx --output output/demo/imputed_sample.xlsx
```

## 2. Start the API

Run `start_malco_demo.bat` or `bash start_malco_demo.sh`. The default URL is http://127.0.0.1:3000. The landing page is informational; the reproducibility workflow uses API requests.

From another terminal at the repository root (use `curl.exe` in Windows PowerShell):

```bash
curl http://127.0.0.1:3000/api/health
curl -X POST http://127.0.0.1:3000/api/submit-answers -H "Content-Type: application/json" --data-binary @output/demo/input/patient_request.json
curl "http://127.0.0.1:3000/api/chat?ref=demo001"
curl -N -X POST "http://127.0.0.1:3000/api/chat?ref=demo001" -H "Content-Type: application/json" --data-binary @data/demo_message.json
curl -X POST "http://127.0.0.1:3000/api/review?ref=demo001" -H "Content-Type: application/json" --data-binary "{}"
```

Use a fresh `referenceKey` for each new intake; duplicates return 409. Chat messages arrive as SSE `data:` frames with `type: delta`, followed by `done` or `error`. Predictions include numerical `shap.contributions` sent to the conversation context. Review returns its mode and `review_complete`; demo results explicitly say no clinical safety assessment was performed. Failed live review requests return 502 with `Need Physician Review`.

Session files live in `output/demo/sessions/<ref>/`; only that exact session's artifacts are copied into `output/demo/archive/<ref>/`. The original session remains readable.

Review closes a session to further chat messages (409); use a new session for a new conversation. Live responses must finish with `finish_reason: stop`. Truncated/filtered/incompletely terminated chat replies produce an SSE error and are not saved; truncated reviews return 502 and cannot pass.

## 3. Run R examples

Install R >=4.3, then restore the supplied dependency snapshot using the command in `statistical_analysis/setup.R` (R 4.5.2 was tested). If using an existing R library, set `R_LIBS_USER` to that library as usual.

From the repository root:

```bash
Rscript statistical_analysis/DeLong_Test_analysis.R
Rscript statistical_analysis/ROC_with_survival_analysis.R
Rscript statistical_analysis/questionnaire_results_analysis.R
Rscript statistical_analysis/review_grade_ICC_analysis.R
Rscript statistical_analysis/ablation_study_analysis.R
Rscript statistical_analysis/ablation_study_analysis.R --design four-stage
Rscript tests/statistics.R
```

Expected outputs are under `output/demo/statistics/`: AUC comparison, ROC/KM plots, questionnaire plots, ICC workbook/plot, and the two-condition ablation plot plus `paired_tests.csv`. R scripts also print `sessionInfo()`.

Use `--input`, `--data-dir`, `--internal-dir`, `--external-dir`, `--output-dir`/`--output` as applicable to supply other data. ROC uses binary outcomes and positive-class probabilities, with a fixed direction, and does not filter patients using follow-up data. The ROC/KM script defaults to `--task 5yrM --event-column True_Label --time-column Days_to_5yrM --ci-method delong`; time-column defaults follow the selected task. Verify that event/time fields describe the same endpoint and censoring horizon. Overall-mortality analysis requires explicitly matched event/time columns; `Death.24` is not assumed to match every endpoint. Use `--km false` for ROC alone. `--ci-method bootstrap` explicitly selects bootstrap intervals.

The default ablation design compares two conditions across five metrics: five tests. `--design four-stage` compares all six V1-V4 pairs across five metrics: 30 tests, corrected together before selecting the 15 adjacent comparisons for figure labels. Both designs require complete ratings from two raters of the same patients and average them within each patient/model/metric. Tests are two-sided paired Wilcoxon with a normal approximation, continuity correction, and no rounding before Bonferroni correction; identical pairs yield P=1. The plotted stars come only from `p.adj.signif` in `paired_tests.csv` (four-star convention documented in the figure). `mean_scores.csv` and `adjacent_tests.csv` are also exported.

The four-stage example reads generated `output/demo/input/ablation_four_stage.csv`. For your own ratings use `--design four-stage --input /path/to/ratings.csv --output-dir /path/to/results`; see DATA_FORMAT. Generated inputs are software fixtures and do not reproduce the manuscript's numerical results. Real study ratings are not bundled.

## 4. Live model configuration (optional)

Copy `.env.example` to `malco_framework/.env.local`. Configure:

- `MALCO_MODE=live`
- `CHAT_BASE_URL`, `CHAT_API_KEY`, `CHAT_MODEL`
- `REVIEW_BASE_URL`, `REVIEW_API_KEY` and provider-supported model identifiers in `REVIEW_MODELS` or `REVIEW_CONFIGURATIONS`
- Optional `PYTHON_EXECUTABLE`, `MALCO_MODELS_DIR`, `MALCO_TASKS` and `MALCO_OUTPUT_DIR`

All URLs must be actual OpenAI-compatible endpoints. Per-reviewer providers can use an `envPrefix` to select their own `<prefix>_BASE_URL` and `<prefix>_API_KEY`. The default timeout is 60 seconds per LLM call; Python inference has a 120-second limit. Offline and localhost-test success does not verify a real provider's availability.

## 5. Prepare upload files

`python scripts/package_release.py --output GitHub_Upload` creates a complete code-only release using an explicit allowlist and hash manifest. It excludes source manuscripts, real validation data, secrets, environments, dependencies, generated outputs and model weights. Review this folder before manually publishing; the packaging command does not contact GitHub.
