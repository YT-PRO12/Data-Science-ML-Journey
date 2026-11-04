# Machine-learning experiment review checklist

A reproducible notebook should separate data exploration, model fitting, validation, and final evaluation.

## Split and preprocessing

- Separate a held-out test set before tuning thresholds or comparing candidate models.
- Fit imputers, scalers, encoders, feature selectors, and oversamplers **only** on training data.
- For time-series or grouped records, use time-aware or group-aware splitting instead of a random row split when appropriate.
- Inspect duplicates, class distribution, and subject leakage between splits.
- Record the random seed, library versions, dataset source, and all selection decisions.

## Report the right evidence

For classification, include a confusion matrix and per-class precision, recall, and F1. State the positive class, support counts, and averaging method. For regression, explain units and the meaning of MAE/RMSE. Compare results against a simple baseline.

Do not use the test set repeatedly for model choice; if a chosen model is evaluated on a test set used during selection, note that the estimate may be optimistic. Where stakes are high, calibration, subgroup behavior, uncertainty, and external validation require independent examination.

A passing notebook or visually plausible plot alone is not proof of model validity in deployment.
