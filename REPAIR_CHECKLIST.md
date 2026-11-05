# Reviewer 1: issue-by-issue repair record

The working source folder had no `.git` metadata. Existing source was backed up before changes. The original public synthetic workbook is preserved byte-for-byte. Research validation data and manuscript files are excluded from this work and the upload package.

| Audit item | Repair | Verification |
|---|---|---|
| F01 API entry point | Complete Next.js app and registered API routes | Production build; localhost HTTP workflow |
| F02 Input placeholders | CLI arguments, repository-relative paths, CSV/XLSX, shared 41-predictor allowlist | Full synthetic workflow; input validation tests |
| F03 Missing model/scaler chain | Saved XGBoost JSON + metadata; matching unscaled training and inference | Eight reload/evaluation tasks; model feature checks |
| F04 Paths and filenames | Shared output directories, task names and plural prediction filenames | Python outputs consumed directly by R |
| F05 ref mismatch | Consistent query ref for chat/review, optional awaited dynamic params | Documented URL tested through HTTP |
| F06 Dependencies | Python exact dependencies, npm lock, R package snapshot and restore helper | Fresh Python/Node installation; existing R library tested; fresh R restore pending |
| F07 False success | Missing data/models cause nonzero exits; review errors return 502 and cannot pass | Negative-path tests |
| F08 Missing R examples | Explicitly synthetic generators for all five R scripts | All five scripts execute and produce outputs |
| F09 DeLong dependency | dplyr::case_when, uniform columns, stop on failed/empty result | Eight-task AUC workbook |
| F10 LLM configuration | .env.local template, checked URLs/credentials, configurable models | Offline mode and localhost adapter tests; real provider untested |
| F11 SHAP context | Actual feature values/contributions/base values in log-odds sent in messages | Additivity and request-content tests |
| F12 Streaming | Valid SSE delta/done/error events | Event parser and HTTP test |
| F13 Empty imputation | No-op for complete data; outcome exclusion, bounded workers, failure propagation | No-missing regression test and sample imputation |
| F14 Review verdicts | Exact documented verdict parsing; error/unknown fail closed | Pass/veto/request-failure tests |
| F15 Miscellaneous | JSON-only Python stdout, exact per-session archive, configurable Python path, measured timeout allowance | Malformed input and archive isolation checks |

Additional corrections: model JSON is saved/loaded through Python byte I/O for Windows Unicode paths; R scripts establish a UTF-8 locale before resolving paths. Ablation stars derive only from full-precision Bonferroni-adjusted p-values exported in the results table. Two-condition analysis uses five tests; the added four-stage analysis uses all 30 comparisons before selecting adjacent pairs. Software demonstration results do not establish manuscript results.

## Second-pass checks, 2026-11-05

- Reproduced and fixed different patient sets across rater files being silently pooled. Complete two-rater panels are required; pairing explicitly uses patient IDs.
- Unified the star field and removed rounding before correction. Added four-stage input, 30-test export and 15 adjacent comparisons; synthetic examples and statistical regression checks included.
- Reproduced and fixed ROC requiring internal survival fields. ROC now uses all valid binary prediction rows, independently of KM eligibility. KM validates external event/time/class columns and defaults to the same endpoint rather than assuming Death.24 for every task.
- Reproduced automatic ROC direction reporting AUC=1 for reversed probabilities whose positive-class AUC is 0. Direction is now fixed to the probability of class 1.
- Reproduced and fixed truncated chat/review responses counting as complete; explicit normal completion is required.
- Reproduced and fixed appending unreviewed chat after a stored review. Reviewed sessions are closed to further chat and remain readable.
- Reproduced and fixed null/malformed JSON returning 500 instead of 400, and direct artifact loading via the documented rehospitalization alias failing.
- Executed a one-candidate non-demo parameter search on generated synthetic data: saving and reloading work. A suspected serialization failure did not reproduce and was not treated as a defect.

The runnable demonstration uses native XGBoost missing handling; standalone ensemble imputation is separately callable and never includes target/time columns. The historical research preprocessing and results require independent evidence. Full non-demo hyperparameter ranges and live A/B review-selection probabilities are retained; small demo search settings and fixed LLM test doubles are explicitly labelled.

All default model weights, patient inputs, survey inputs and rating examples are newly generated synthetic artifacts for software verification. They are not clinical results or genuine physician evaluations.
