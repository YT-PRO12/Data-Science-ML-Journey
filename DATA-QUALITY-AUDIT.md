# Notebook data-quality audit

Before drawing conclusions from any table or training a model:

1. Record source, license, extraction date, schema, row count, and relevant units.
2. Inspect missingness per field and duplicated records or identifiers.
3. Check expected value ranges, impossible dates, class balance, and category drift.
4. For supervised learning, ensure the target or post-outcome features cannot leak into predictors.
5. Separate train/test data **before** fitting transformations or tuning thresholds.
6. For temporal or grouped data, use splits that respect time and entity identity.
7. Run notebooks from a clean kernel and preserve package versions and random seeds.

Document each cleaning decision and its effect on sample counts. A visually convincing plot does not establish causal inference or external validity.

See [ML evaluation checklist](ML_EVALUATION_CHECKLIST.md).
