# Usage guide

## 1. Run the demo

From the repository root, run `setup.bat`, `run_ml_pipeline.bat`, then `start_malco_demo.bat`. In PowerShell, prefix each with `.\`; on Linux/macOS, use `bash` with the corresponding `.sh` files.

Outputs go to `output/demo/`: eight saved models, validation predictions, SHAP explanations and `demo_summary.json` with `completed_tasks: 8`. All example inputs and models are synthetic.

For Python commands below, use `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux/macOS instead of `python`.

## 2. Try the API

Keep the service running. In another terminal at the repository root, run these requests in order (use `curl.exe` in PowerShell):

```bash
curl http://127.0.0.1:3000/api/health
curl -X POST http://127.0.0.1:3000/api/submit-answers -H "Content-Type: application/json" --data-binary '@output/demo/input/patient_request.json'
curl "http://127.0.0.1:3000/api/chat?ref=demo001"
curl -N -X POST "http://127.0.0.1:3000/api/chat?ref=demo001" -H "Content-Type: application/json" --data-binary '@data/demo_message.json'
curl -X POST "http://127.0.0.1:3000/api/review?ref=demo001" -H "Content-Type: application/json" --data-binary '{}'
```

Use a new `referenceKey` for each intake. Chat streams `delta` then `done` or `error`; review reports `review_complete`. Duplicate keys or chatting after review return 409; incomplete live reviews return 502 and cannot pass. Session records are in `output/demo/sessions/`, with reviewed copies in `output/demo/archive/`.

## 3. Run statistics

Use an existing compatible R library or restore dependencies with `Rscript statistical_analysis/setup.R`. Then run:

```bash
Rscript statistical_analysis/DeLong_Test_analysis.R
Rscript statistical_analysis/ROC_with_survival_analysis.R
Rscript statistical_analysis/questionnaire_results_analysis.R
Rscript statistical_analysis/review_grade_ICC_analysis.R
Rscript statistical_analysis/ablation_study_analysis.R
Rscript statistical_analysis/ablation_study_analysis.R --design four-stage
```

Results go to `output/demo/statistics/`. For your own data, use the scripts' `--input`/`--data-dir` and `--output-dir` options; see [Data format](data/DATA_FORMAT.md). The ROC/KM script accepts `--internal-dir` and `--external-dir`; KM requires matched event/time fields (`--event-column`, `--time-column`), or use `--km false` for ROC only.

Ablation uses two-sided paired Wilcoxon tests and Bonferroni correction across five comparisons (two conditions) or 30 (four stages). The CSV results and plot stars use the same adjusted P values.

## 4. Optional live LLMs and checks

Copy [.env.example](.env.example) to `malco_framework/.env.local`; set `MALCO_MODE=live`, `CHAT_BASE_URL`, `CHAT_API_KEY`, `CHAT_MODEL`, `REVIEW_BASE_URL`, `REVIEW_API_KEY` and three supported models in `REVIEW_MODELS`. Use OpenAI-compatible endpoints. Real-provider availability is not covered by offline testing.

```bash
python -m unittest discover -s tests -p test_pipeline.py -v
node --test tests/api.test.mjs
Rscript tests/statistics.R
npm --prefix malco_framework run build
```

The optional imputation check requires first running `python ml_prognostic_models/ensemble_imputation.py --input data/synthetic_sample.xlsx --output output/demo/imputed_sample.xlsx`. See [Validation](VALIDATION.txt) for tested environments and remaining limits, and [Packaging](FILE_CHECKLIST.md) to prepare an upload folder.
