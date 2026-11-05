# Core code fixes

| Area | Changes |
|---|---|
| Runnable workflow | Working API routes, configurable paths, locked dependencies and complete synthetic examples. |
| Model inputs | One 41-predictor schema; outcomes and time-to-event fields excluded; missing-input validation. |
| Model artifacts | Saved JSON models and metadata; consistent unscaled training/inference; matching task names and prediction files. |
| Statistical analysis | Complete ID-matched rating panels; full-precision Bonferroni P values for tables and stars (5 or 30 tests); fixed ROC direction; separate KM event/time validation. |
| Agent workflow | Numerical SHAP context, valid streaming events, exact review verdicts and session-specific archives. |
| Failure handling | Missing inputs fail; incomplete replies cannot pass; reviewed sessions reject further chat; malformed requests return errors. |
| Compatibility | Windows Unicode paths, task aliases and complete-data imputation handled. |

See [Validation](VALIDATION.txt) for checks and remaining limits. Demonstration models and ratings are synthetic; these software fixes do not establish historical clinical results.
