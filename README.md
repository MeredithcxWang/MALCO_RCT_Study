
# MALCO_RCT Study

**MALCO** (Multi-Agent Large Language Model for Cardiac Outcomes) is an integrated AI framework for personalized Heart Failure care. This project provides the source code for the manuscript "Multi-agent Large Language Model Framework Integrating Prognostic Machine Learning for Personalized Heart Failure Management: a Multi-center Randomized Controlled Trial".
## Prerequisite

### 1. Python (≥ 3.9)
#### Python Packages

```bash
pip install numpy pandas scikit-learn xgboost scipy shap matplotlib joblib psutil
```

| Package | Tested Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | Numerical computation |
| `pandas` | ≥ 2.0 | Data manipulation |
| `scikit-learn` | ≥ 1.3 | Machine learning (IterativeImputer, KNN, BayesianRidge, ExtraTrees, PCA, StandardScaler, model evaluation) |
| `xgboost` | ≥ 2.0 | Gradient-boosted tree classifier |
| `scipy` | ≥ 1.11 | Statistical distributions |
| `shap` | ≥ 0.44 | Model interpretability (SHAP values) |
| `matplotlib` | ≥ 3.7 | Plotting (ROC curves, SHAP plots) |
| `joblib` | ≥ 1.3 | Model serialisation / persistence |
| `psutil` | ≥ 5.9 | System resource monitoring |

> **Note:** `warnings`, `collections`, `sys`, `os`, and `concurrent.futures` are part of the Python standard library and require no additional installation.

---

### 2. Node.js (≥ 18.0.0)
#### JavaScript / Next.js Dependencies

After cloning the repository, install all Node packages:

```bash
cd web          # or whichever directory contains package.json
npm install
```

| Package | Version | Purpose |
|---|---|---|
| `next` | ^15.1.0 | React framework (SSR / SSG) |
| `react` | ^19.0.0 | UI library |
| `react-dom` | ^19.0.0 | DOM rendering for React |
| `openai` | ^4.78.0 | OpenAI API client |
| `eslint` | ^9.17.0 | Code linting (dev) |
| `eslint-config-next` | ^15.1.0 | Next.js ESLint rules (dev) |

---

### 3. R (≥ 4.3)
#### R Packages

Run the following in an R console to install all required packages at once:

```r
install.packages(c(
  "tidyverse",
  "readxl",
  "openxlsx",
  "ggsci",
  "rstatix",
  "pROC",
  "ggplot2",
  "dplyr",
  "ggpubr",
  "irr",
  "scales",
  "survival",
  "survminer"
))
```

| Package | Purpose |
|---|---|
| `tidyverse` | Core data-science toolkit (dplyr, ggplot2, tidyr, etc.) |
| `readxl` | Read Excel files (.xlsx / .xls) |
| `openxlsx` | Write Excel files |
| `ggsci` | Scientific journal colour palettes |
| `rstatix` | Pipe-friendly statistical tests |
| `pROC` | ROC curve analysis & AUC computation |
| `ggplot2` | Data visualisation |
| `dplyr` | Data manipulation |
| `ggpubr` | Publication-ready plots |
| `irr` | Inter-rater reliability (e.g., Cohen's κ) |
| `scales` | Axis / colour scale helpers |
| `survival` | Survival analysis (Cox models, Kaplan–Meier) |
| `survminer` | Survival curve visualisation |

> **Note:** `grid` ships with base R and requires no additional installation.

## File Descriptions
### `data/` — Patient Data Directory

This directory contains sample data and format documentation for the MALCO framework. Due to IRB restrictions and patient privacy regulations, **no actual patient data is included** in this repository.

| File | Description |
|------|-------------|
| `synthetic_sample.xlsx` | **Synthetic sample dataset** containing 100 synthetically generated patient profiles for demonstration and testing purposes. Each row represents one patient with 41 clinical features + 2 outcome variables (e.g., `Days_to_2yrM`, `2yrM`). These profiles were generated using statistical distributions derived from the real cohort but **do NOT represent actual patients**. This file demonstrates the exact format expected by the ML pipeline and MALCO framework. |
| `DATA_FORMAT.md` | **Comprehensive data format specification** documenting:<br>• All 41 clinical features across 5 domains (demographics, lifestyle/comorbidities, pharmacotherapy, laboratory parameters)<br>• Column naming conventions and data types<br>• Missing data handling strategies<br>• Integration with ML pipeline and MALCO framework<br>• Validation checklist for preparing new datasets<br>This document serves as the authoritative reference for data preparation and is essential reading for anyone attempting to reproduce the pipeline or adapt it to new heart failure cohorts. |

**For reproducibility testing:** Users with access to similar heart failure cohort data can place their datasets in this directory following the format specified in `DATA_FORMAT.md`. The ML pipeline scripts (`ensemble_imputation.py`, `xgboost_training.py`, `model_evaluation.py`) expect data files to match this schema.


### `ml_prognostic_models/` — Machine Learning Prognostic Models

Scripts for developing and validating the XGBoost-based prognostic models on a Hong Kong heart failure cohort (N=241,982).

| File | Description |
|------|-------------|
| `ensemble_imputation.py` | Custom ensemble imputation pipeline for handling missing values in the EHR dataset.|
| `xgboost_training.py` | XGBoost classifier training script |
| `model_evaluation.py` | External validation script that loads pre-trained XGBoost models and scalers, generates predicted probabilities on independent external validation datasets, computes AUROC, plots ROC curves, and exports prediction results.|

### `statistical_analysis/` — Statistical Analysis 

R scripts for analyzing the RCT outcomes, model performance, and expert adjudication results.

| File | Description |
|------|-------------|
| `ROC_with_Survival_Analysis.R` | Generates ROC curves with bootstrapped 95% CI AUC for both internal and external validation cohorts, and Kaplan-Meier overall survival curves for the external validation cohort. |
| `DeLong_Test_Analysis.R` | Computes AUC with 95% DeLong confidence intervals for internal and external validation cohorts across all prediction tasks, performs unpaired DeLong tests to compare discriminative performance between cohorts. |
| `Questionnaire_Results_Analysis.R` | Analyzes pre- vs. post-intervention survey scores from the RCT. Performs Wilcoxon rank-sum tests and generates violin-boxplot comparison figures for specified survey items. |
| `Review_Grade_ICC_Analysis.R` | Calculates two-way random, absolute agreement, average-measures intraclass correlation coefficients (ICC(2,k)) for each of the seven expert evaluation metrics across four raters. |
| `Ablation_Study_Analysis.R` | Compares Likert-scale evaluation scores (1–5) between the full MALCO framework and the baseline LLM across five quality metrics. Performs paired Wilcoxon signed-rank tests. |

### `malco_framework/` — MALCO Multi-Agent Framework

This directory contains the **complete pipeline framework** for the MALCO system, implementing three collaborative AI agents that integrate prognostic machine learning with large language models for personalized heart failure care.

#### System Architecture

The MALCO framework operates as a **sequential three-agent pipeline**:

1. **Patient Portfolio Agent** → Intake & ML Prediction
2. **Conversational Agent** → Patient-AI Dialogue with SHAP-informed Context
3. **Review Panel Agent** → Multi-LLM Safety Audit with One-Vote Veto

Each agent is implemented as a Next.js API route handler, with the Patient Portfolio Agent invoking a Python subprocess for XGBoost inference and SHAP explainability.

---

#### Core Agent Files

| File | Description |
|------|-------------|
| `patient_portfolio_agent.js` | **Patient Portfolio Agent (Agent 1)** — Implemented as a Next.js API route (`POST /api/submit-answers`).<br><br>**Function:**<br>• Receives 41 clinical features from patient questionnaire (questions Q1–Q41)<br>• Maps questionnaire responses to XGBoost model feature names via `FEATURE_MAP`<br>• Persists patient data to master CSV (`data.csv`) for audit trail<br>• Generates single-row inference CSV and invokes Python subprocess: `python run_models.py <csv_path> <referenceKey>`<br>• Parses stdout from Python to extract 8 risk predictions and risk tiers<br>• Returns structured JSON: `{ predictions: [{model, value, risk}, ...] }`<br><br>**Key Design:**<br>• Rule-based orchestrator (no LLM)<br>• CSV sanitization prevents formula injection<br>• Reference key validation prevents path traversal|
| `conversational_agent.js` | **Conversational Agent (Agent 2)** — Implemented as Next.js API routes (`GET/POST /api/chat`).<br><br>**Function:**<br>• `GET`: Initializes conversation by loading patient CSV, prediction results, and SHAP force plot references into a structured prompt<br>• `POST`: Accepts user message, appends to chat history, calls Gemini-2.5-pro via OpenAI-compatible API, streams response via Server-Sent Events<br>• Persists full chat history to `<ref>_chat.json` and plain-text log to `<ref>_chatlog.txt`<br><br>**Core Prompts:**<br>• `SYSTEM_IDENTITY`: Defines agent as a health assistant with medical background, safety boundaries (no unverified advice, privacy-preserving), and communication style (accurate, evidence-based, accessible language)<br>• `INITIAL_TASK`: Instructs agent to analyze patient data + ML predictions + SHAP plots, provide risk communication (implicit mortality analysis avoiding "death" terminology), and actionable recommendations<br>• Temperature set to 0.2 for reproducibility<br><br>**SHAP Integration:**<br>• References SHAP force plots by filename: `<ref>_<model>_force.png` |
| `review_panel_agent.js` | **Review Panel Agent (Agent 3)** — Implemented as a Next.js API route (`POST /api/review`).<br><br>**Function:**<br>• Loads full chat log from `<ref>_chatlog.txt`<br>• Randomly selects one of two LLM configurations via weighted sampling:<br>&nbsp;&nbsp;◦ **Configuration A (40%):** DeepSeek-R1, o3, Claude 3.7 Sonnet<br>&nbsp;&nbsp;◦ **Configuration B (60%):** GPT-5, Grok-4, Claude Opus 4.1<br>• Sends audit prompt to all 3 LLMs in selected config **concurrently** (via `Promise.all`)<br>• Each LLM independently audits chat log for:<br>&nbsp;&nbsp;◦ Medical errors or inaccuracies<br>&nbsp;&nbsp;◦ Potentially misleading information<br>&nbsp;&nbsp;◦ Completeness gaps or omissions<br>&nbsp;&nbsp;◦ Scope violations (e.g., prescribing medications)<br>• Parses first line of each LLM response for verdict: `"Passed"` or `"Requires Doctor Intervention"`<br>• **One-Vote Veto Protocol:** If any LLM flags "need_doctor" or "unknown", final decision is `"Need Physician Review"`<br>• Archives all session files (`<ref>*`) to `ML/history_patients/<ref>/` directory<br><br>**Audit Prompt Design:**<br>• Explicitly instructs reviewers: "Patient has been informed they're conversing with AI"<br>• Priority hierarchy: detecting misinformation > omission of reminders<br>• Requires verdict as first line: `"Passed"` or `"Requires Doctor Intervention"` followed by justification<br>• Temperature set to 0.2 for consistent safety thresholds |
| `run_models.py` | **Python Inference Engine** — Invoked as subprocess by Patient Portfolio Agent.<br><br>**Function:**<br>• Loads 8 pre-trained XGBoost models and scalers from `ML/models/` directory<br>• For each model (2/3/4/5-year mortality, cancer, COPD, pneumonia, rehospitalization):<br>&nbsp;&nbsp;◦ Scales input features using model-specific `StandardScaler`<br>&nbsp;&nbsp;◦ Predicts risk probability via `model.predict_proba()`<br>&nbsp;&nbsp;◦ Applies threshold-based risk stratification relative to Youden's Index:<br>&nbsp;&nbsp;&nbsp;&nbsp;▪ **Low:** prob ≤ 0.5 × threshold<br>&nbsp;&nbsp;&nbsp;&nbsp;▪ **Low-Medium:** 0.5 × threshold < prob ≤ 0.8 × threshold<br>&nbsp;&nbsp;&nbsp;&nbsp;▪ **Medium-High:** 0.8 × threshold < prob ≤ 1.2 × threshold<br>&nbsp;&nbsp;&nbsp;&nbsp;▪ **High:** prob > 1.2 × threshold<br>&nbsp;&nbsp;◦ Generates SHAP force plot saved as `<ref>_<model>_force.png`<br>• Writes results to `<ref>_results.txt` (CSV format) and stdout for Node.js consumer<br>• Handles missing features by filling NaN <br>|

---

#### Configuration Files

| File | Description |
|------|-------------|
| `package.json` | Node.js project manifest declaring Next.js 15, React 19, and OpenAI SDK as core dependencies. |


---

**API Workflow:**
1. **Patient Intake** → `POST /api/submit-answers` with questionnaire JSON
   - Returns: `{ predictions: [{model, value, risk}, ...] }`
2. **Initialize Chat** → `GET /api/chat?ref=<referenceKey>`
   - Returns: Initial chat history with system prompt + patient context
3. **Send Message** → `POST /api/chat` with `{ message: "..." }`
   - Returns: Streaming response from Gemini-2.5-pro
4. **Audit Chat** → `POST /api/review`
   - Returns: `{ final_decision: "Pass" | "Need Physician Review", applied_configuration: "Configuration_A" | "Configuration_B", agent_results: {...} }`

**Note:** The frontend UI is **not included** in this repository. The API routes can be tested via curl, Postman, or a custom client implementation.

---


### `statistical_analysis/` — Statistical Analysis 

R scripts for analyzing the RCT outcomes, model performance, and expert adjudication results.

| File | Description |
|------|-------------|
| `ROC_with_Survival_Analysis.R` | Generates ROC curves with bootstrapped 95% CI AUC for both internal and external validation cohorts, and Kaplan-Meier overall survival curves for the external validation cohort. |
| `DeLong_Test_Analysis.R` | Computes AUC with 95% DeLong confidence intervals for internal and external validation cohorts across all prediction tasks, performs unpaired DeLong tests to compare discriminative performance between cohorts. |
| `Questionnaire_Results_Analysis.R` | Analyzes pre- vs. post-intervention survey scores from the RCT. Performs Wilcoxon rank-sum tests and generates violin-boxplot comparison figures for specified survey items. |
| `Review_Grade_ICC_Analysis.R` | Calculates two-way random, absolute agreement, average-measures intraclass correlation coefficients (ICC(2,k)) for each of the seven expert evaluation metrics across four raters. |
| `Ablation_Study_Analysis.R` | Compares Likert-scale evaluation scores (1–5) between the full MALCO framework and the baseline LLM across five quality metrics. Performs paired Wilcoxon signed-rank tests. |
