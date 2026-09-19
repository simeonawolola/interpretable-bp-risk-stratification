# Interpretable Risk Stratification for Elevated Blood Pressure in a Diabetes Cohort

Reproducible technical demonstration applying interpretable machine-learning
methods (logistic regression, gradient boosting with permutation importance
and partial dependence analysis) and subgroup performance evaluation to a
public clinical dataset.

Prepared by **Olatunde Simeon Awolola** as a supplementary technical project
in support of a proposed U.S. endeavor as a Health Data Scientist, focused on
responsible and interpretable AI for heterogeneous health data.

## Why this project

The project mirrors, on fully public data, the design of the author's
ongoing retrospective research on predictors of hypertension among patients
with Type 2 diabetes at the School of Health Information Management,
University College Hospital, Ibadan — **no UCH data, patient records, or
protected health information of any kind is used here.**

## Data

`sklearn.datasets.load_diabetes` — 442 patients, 10 baseline clinical
variables (Efron, Hastie, Johnstone & Tibshirani, *Least Angle Regression*,
Annals of Statistics, 2004). Public, de-identified, bundled with
scikit-learn under a permissive open license.

## What's here

- `01_data_and_model.py` — data prep, modeling, interpretability, and subgroup evaluation
- `roc_curves.png`, `logreg_coefficients.png`, `gbc_permutation_importance.png`, `partial_dependence.png` — generated figures
- `Technical_Project_Report.docx` — full technical report with discussion

## Method summary

- **Outcome:** elevated blood pressure, defined as the upper tertile of the
  dataset's blood-pressure variable (a sample-relative research threshold,
  not a clinical diagnostic criterion).
- **Models:** logistic regression (directly interpretable) and gradient
  boosting (explained post hoc).
- **Interpretability:** logistic-regression coefficients, permutation
  importance, and partial dependence plots — standard scikit-learn
  interpretability tools, used in place of the SHAP library, which was not
  available in the environment this project was built in.
- **Responsible-AI check:** model performance evaluated separately by sex,
  with the dataset's size and homogeneity limitations reported openly
  rather than glossed over.

## Results (headline)

| Model | Test AUC | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.798 | 77.5% |
| Gradient Boosting | 0.787 | 73.9% |

Full analysis, figures, and discussion: see `report/Technical_Project_Report.docx`.

## Limitations

This is a methodological demonstration, not a validated clinical tool. It
has not been tested on an independent cohort, has not undergone clinical
validation, and must not be used for individual patient risk assessment.
