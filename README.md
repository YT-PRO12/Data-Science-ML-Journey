# Data Science & Machine Learning Journey

A hands-on learning repository progressing from Python programming to data preparation, visualization, statistics, and machine-learning experiments.

## Learning roadmap

| Stage | Study materials |
| --- | --- |
| 1. Python foundations | [Python](01-Python/) — basics, functions, file handling, and OOP |
| 2. Numerical computing | [NumPy](02-NumPy/) — numerical arrays and practice notebooks |
| 3. Tabular analysis | [Pandas](03-Pandas/) — working with structured datasets |
| 4. Communicating findings | [Data Visualization](04-Data-Visualization/) |
| 5. Statistical reasoning | [Statistics](05-Statistics/) |
| 6. Model building | [Machine Learning](06-Machine%20Learning/) |
| 7. End-to-end practice | [Heart disease classification project](Project-01-Heart-Disease-Classification/) |
| Supporting coursework | [Certifications](CERTIFICATION/) |

## Running the notebooks

Use Python 3.10+ in an isolated virtual environment and install packages required by the notebook being studied (typically `jupyter`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn`). Open notebooks in Jupyter or VS Code, inspect local dataset paths, and run cells in order from a restarted kernel.

Do not assume that every linked external dataset is included in the repository. Save a record of dependency versions and the data snapshot when reproducing results.

## How to evaluate ML projects responsibly

Separate training, validation, and test data; fit all preprocessing steps using training data only. Compare a simple baseline before using a more complex estimator. Report class balance, precision, recall, and limitations rather than only accuracy. A notebook result is not evidence of real-world clinical validity or production deployment.

This repository documents a learning process; individual notebook files are not presented as verified production systems.
