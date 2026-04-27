
# MALCO_RCT Study

**MALCO** (Multi-Agent Large Language Model for Cardiac Outcomes) is an integrated AI framework for personalized Heart Failure care. This project provides the source code for the manuscript "Multi-agent Large Language Model Framework Integrating Prognostic Machine Learning for Personalized Heart Failure Management: a Multi-center Randomized Controlled Trial".
## Prerequisite

### 1. Python (≥ 3.9)

Install Python from [python.org](https://www.python.org/) or via conda.

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

Install Node.js from [nodejs.org](https://nodejs.org/).

```bash
node --version   # verify ≥ 18.0.0
```

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

Install R from [CRAN](https://cran.r-project.org/).  
[RStudio](https://posit.co/download/rstudio-desktop/) is recommended.

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
### `ml_prognostic_models/` — Machine Learning Prognostic Models

Scripts for developing and validating the XGBoost-based prognostic models on a Hong Kong heart failure cohort (N=241,982).

| File | Description |
|------|-------------|
| `ensemble_imputation.py` | Custom ensemble imputation pipeline for handling missing values in the EHR dataset.|
| `xgboost_training.py` | XGBoost classifier training script |
| `model_evaluation.py` | External validation script that loads pre-trained XGBoost models and scalers, generates predicted probabilities on independent external validation datasets, computes AUROC, plots ROC curves, and exports prediction results.|

### `malco_framework/` — MALCO Multi-Agent Framework

Implementation of the three-agent MALCO system: Patient Portfolio Agent, Conversational Agent, and Review Panel Agent. Built as a Next.js web application.

| File | Description |
|------|-------------|
| `package.json` | Node.js project manifest for the MALCO web application. Declares Next.js 15, React 19, and the OpenAI SDK as core dependencies. |
| `patient_portfolio_agent.js` | **Patient Portfolio Agent** — A rule-based orchestrator implemented as a Next.js API route. Handles patient questionnaire submission (41 clinical features mapped to XGBoost model feature names), persists patient data to a master CSV, generates a single-row inference CSV, and invokes the Python prediction pipeline (`run_models.py`). Returns structured risk predictions to the frontend. |
| `conversational_agent.js` | **Conversational Agent** — A prompt-engineered LLM module (Gemini-2.5-pro) implemented as a Next.js API route. Initializes conversation context by assembling patient baseline data, ML risk predictions, and SHAP force plot references into a structured prompt. Provides streaming responses via API. |
| `3_review_panel_agent.js` | **Review Panel Agent** — An ensemble safety auditing module implemented as a Next.js API route. Deploys one of two LLM configurations selected via weighted randomization (40% Configuration A: DeepSeek-R1, o3, Claude 3.7 Sonnet; 60% Configuration B: GPT-5, Grok-4, Claude Opus 4.1). Each model independently audits the full patient–AI chat log for clinical inaccuracies, misleading information, completeness gaps, and scope violations. Implements a **one-vote veto protocol**: any single "Need Physician Review" verdict triggers clinician review. Archives all patient session files to a history directory upon completion. |
| `run_models.py` | Python inference engine called by the Patient Portfolio Agent. Loads eight pre-trained XGBoost models and scalers, predicts risk probabilities for a single patient, applies threshold-based risk stratification, generates per-model SHAP force plots for explainability, and outputs results to both a text file and stdout for the Node.js consumer. |

### `statistical_analysis/` — Statistical Analysis 

R scripts for analyzing the RCT outcomes, model performance, and expert adjudication results.

| File | Description |
|------|-------------|
| `ROC_with_Survival_Analysis.R` | Generates ROC curves with bootstrapped 95% CI AUC for both internal and external validation cohorts, and Kaplan-Meier overall survival curves for the external validation cohort. |
| `DeLong_Test_Analysis.R` | Computes AUC with 95% DeLong confidence intervals for internal and external validation cohorts across all prediction tasks, performs unpaired DeLong tests to compare discriminative performance between cohorts. |
| `Questionnaire_Results_Analysis.R` | Analyzes pre- vs. post-intervention survey scores from the RCT. Performs Wilcoxon rank-sum tests and generates violin-boxplot comparison figures for specified survey items. |
| `Review_Grade_ICC_Analysis.R` | Calculates two-way random, absolute agreement, average-measures intraclass correlation coefficients (ICC(2,k)) for each of the seven expert evaluation metrics across four raters. |
| `Ablation_Study_Analysis.R` | Compares Likert-scale evaluation scores (1–5) between the full MALCO framework and the baseline LLM across five quality metrics. Performs paired Wilcoxon signed-rank tests. |
## File Descriptions
### `ml_prognostic_models/` — Machine Learning Prognostic Models

Scripts for developing and validating the XGBoost-based prognostic models on a Hong Kong heart failure cohort (N=241,982).

| File | Description |
|------|-------------|
| `ensemble_imputation.py` | Custom ensemble imputation pipeline for handling missing values in the EHR dataset.|
| `xgboost_training.py` | XGBoost classifier training script |
| `model_evaluation.py` | External validation script that loads pre-trained XGBoost models and scalers, generates predicted probabilities on independent external validation datasets, computes AUROC, plots ROC curves, and exports prediction results.|

### `malco_framework/` — MALCO Multi-Agent Framework

Implementation of the three-agent MALCO system: Patient Portfolio Agent, Conversational Agent, and Review Panel Agent. Built as a Next.js web application.

| File | Description |
|------|-------------|
| `package.json` | Node.js project manifest for the MALCO web application. Declares Next.js 15, React 19, and the OpenAI SDK as core dependencies. |
| `patient_portfolio_agent.js` | **Patient Portfolio Agent** — A rule-based orchestrator implemented as a Next.js API route. Handles patient questionnaire submission (41 clinical features mapped to XGBoost model feature names), persists patient data to a master CSV, generates a single-row inference CSV, and invokes the Python prediction pipeline (`run_models.py`). Returns structured risk predictions to the frontend. |
| `conversational_agent.js` | **Conversational Agent** — A prompt-engineered LLM module (Gemini-2.5-pro) implemented as a Next.js API route. Initializes conversation context by assembling patient baseline data, ML risk predictions, and SHAP force plot references into a structured prompt. Provides streaming responses via API. |
| `3_review_panel_agent.js` | **Review Panel Agent** — An ensemble safety auditing module implemented as a Next.js API route. Deploys one of two LLM configurations selected via weighted randomization (40% Configuration A: DeepSeek-R1, o3, Claude 3.7 Sonnet; 60% Configuration B: GPT-5, Grok-4, Claude Opus 4.1). Each model independently audits the full patient–AI chat log for clinical inaccuracies, misleading information, completeness gaps, and scope violations. Implements a **one-vote veto protocol**: any single "Need Physician Review" verdict triggers clinician review. Archives all patient session files to a history directory upon completion. |
| `run_models.py` | Python inference engine called by the Patient Portfolio Agent. Loads eight pre-trained XGBoost models and scalers, predicts risk probabilities for a single patient, applies threshold-based risk stratification, generates per-model SHAP force plots for explainability, and outputs results to both a text file and stdout for the Node.js consumer. |

### `statistical_analysis/` — Statistical Analysis 

R scripts for analyzing the RCT outcomes, model performance, and expert adjudication results.

| File | Description |
|------|-------------|
| `ROC_with_Survival_Analysis.R` | Generates ROC curves with bootstrapped 95% CI AUC for both internal and external validation cohorts, and Kaplan-Meier overall survival curves for the external validation cohort. |
| `DeLong_Test_Analysis.R` | Computes AUC with 95% DeLong confidence intervals for internal and external validation cohorts across all prediction tasks, performs unpaired DeLong tests to compare discriminative performance between cohorts. |
| `Questionnaire_Results_Analysis.R` | Analyzes pre- vs. post-intervention survey scores from the RCT. Performs Wilcoxon rank-sum tests and generates violin-boxplot comparison figures for specified survey items. |
| `Review_Grade_ICC_Analysis.R` | Calculates two-way random, absolute agreement, average-measures intraclass correlation coefficients (ICC(2,k)) for each of the seven expert evaluation metrics across four raters. |
| `Ablation_Study_Analysis.R` | Compares Likert-scale evaluation scores (1–5) between the full MALCO framework and the baseline LLM across five quality metrics. Performs paired Wilcoxon signed-rank tests. |
