# MALCO

Research code for **Multi-Agent LLM-powered Collaborative Outpatient Care**, including prognostic modelling, multi-agent orchestration and statistical analysis.

## Quick start

Requires Python 3.11–3.13 and Node.js ≥20.9. R ≥4.3 is optional for statistical analyses.

Windows PowerShell:

```powershell
git clone https://github.com/MeredithcxWang/MALCO_RCT_Study.git
cd MALCO_RCT_Study
.\setup.bat
.\run_ml_pipeline.bat
.\start_malco_demo.bat
```

The pipeline trains eight demonstration models and saves predictions and SHAP explanations under `output/demo/`. The service runs at [localhost:3000](http://127.0.0.1:3000); see the usage guide for API requests and Linux/macOS commands.

## Usage scope

- The demonstration uses synthetic data and newly trained models. Clinical data and study model weights are not included.
- Offline chat and review use fixed test responses. Live LLM use requires your own API credentials.
- Windows execution has been validated; other platforms remain untested.

## Documentation

[Usage guide](QUICKSTART.md) · [Data format](data/DATA_FORMAT.md) · [Validation](VALIDATION.txt)
