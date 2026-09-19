"""
Interpretable Risk Stratification for Elevated Blood Pressure
in a Diabetes-Progression Cohort

Author: Olatunde Simeon Awolola
Purpose: Supplementary technical project supporting EB-2 NIW petition
         (interpretable / responsible AI for heterogeneous health data)

Data: scikit-learn's bundled "diabetes" dataset (Efron, Hastie, Johnstone,
      Tibshirani, "Least Angle Regression", Annals of Statistics, 2004).
      442 patients, 10 baseline clinical variables, publicly distributed
      with scikit-learn under a permissive open license. No PHI, no
      UCH data, no identifiable records of any kind.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                              classification_report, RocCurveDisplay)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

RNG = 42
np.random.seed(RNG)

# ---------------------------------------------------------------------
# 1. Load data (real clinical units, not pre-standardized)
# ---------------------------------------------------------------------
raw = load_diabetes(scaled=False)
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["sex"] = df["sex"].map({1.0: "male", 2.0: "female"})  # per dataset codebook
df["progression"] = raw.target

# ---------------------------------------------------------------------
# 2. Define outcome: elevated blood pressure (top tertile of 'bp')
#    This mirrors the petitioner's own retrospective UCH study design
#    ("predictors of hypertension among patients with Type 2 diabetes"),
#    reproduced here on a fully public dataset for methodological
#    demonstration only -- NOT a claim about the UCH patient population.
# ---------------------------------------------------------------------
bp_tertile_cutoff = df["bp"].quantile(2 / 3)
df["elevated_bp"] = (df["bp"] >= bp_tertile_cutoff).astype(int)

predictors = ["age", "sex", "bmi", "s1", "s2", "s3", "s4", "s5", "s6", "progression"]
X = pd.get_dummies(df[predictors], columns=["sex"], drop_first=True)
y = df["elevated_bp"]

print(f"Elevated-BP prevalence in sample: {y.mean():.1%}")
print(f"BP tertile cutoff used to define 'elevated': {bp_tertile_cutoff:.1f} (dataset units)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RNG, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ---------------------------------------------------------------------
# 3. Models: an inherently interpretable baseline (logistic regression)
#    and a higher-capacity model (gradient boosting) explained post hoc.
# ---------------------------------------------------------------------
logreg = LogisticRegression(max_iter=2000, random_state=RNG)
logreg.fit(X_train_s, y_train)

gbc = GradientBoostingClassifier(random_state=RNG, n_estimators=200, max_depth=2, learning_rate=0.05)
gbc.fit(X_train_s, y_train)

results = {}
for name, model in [("Logistic Regression", logreg), ("Gradient Boosting", gbc)]:
    proba = model.predict_proba(X_test_s)[:, 1]
    pred = model.predict(X_test_s)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    results[name] = dict(auc=auc, acc=acc, cm=cm, proba=proba, pred=pred)
    print(f"\n{name}: AUC={auc:.3f}  Accuracy={acc:.3f}")
    print(cm)
    print(classification_report(y_test, pred, digits=3))

# ROC curves
fig, ax = plt.subplots(figsize=(5, 5))
for name, model in [("Logistic Regression", logreg), ("Gradient Boosting", gbc)]:
    RocCurveDisplay.from_estimator(model, X_test_s, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("ROC — Elevated Blood Pressure Classification")
fig.tight_layout()
fig.savefig("figures/roc_curves.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# 4. Interpretability: logistic-regression coefficients (directly
#    interpretable on standardized scale) + permutation importance for
#    the gradient boosting model + partial dependence for top features.
# ---------------------------------------------------------------------
coef_df = pd.Series(logreg.coef_[0], index=X_train_s.columns).sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
coef_df.plot(kind="barh", ax=ax, color=["#c0392b" if v < 0 else "#2166ac" for v in coef_df])
ax.set_title("Logistic Regression Coefficients\n(standardized features; +ve = higher elevated-BP odds)")
ax.set_xlabel("Coefficient")
fig.tight_layout()
fig.savefig("figures/logreg_coefficients.png", dpi=150)
plt.close(fig)

perm = permutation_importance(gbc, X_test_s, y_test, n_repeats=30, random_state=RNG, scoring="roc_auc")
perm_df = pd.Series(perm.importances_mean, index=X_test_s.columns).sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
perm_df.plot(kind="barh", ax=ax, color="#2c7fb8")
ax.set_title("Gradient Boosting — Permutation Importance\n(mean AUC drop, 30 repeats)")
ax.set_xlabel("Importance (Δ AUC)")
fig.tight_layout()
fig.savefig("figures/gbc_permutation_importance.png", dpi=150)
plt.close(fig)

top_features = perm_df.sort_values(ascending=False).index[:4].tolist()
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(gbc, X_test_s, top_features, ax=ax)
fig.suptitle("Partial Dependence — Top 4 Predictors (Gradient Boosting)")
fig.tight_layout()
fig.savefig("figures/partial_dependence.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# 5. Subgroup evaluation (responsible-AI check): does performance hold
#    evenly across sex? Reported honestly, including sample-size caveats.
# ---------------------------------------------------------------------
subgroup_rows = []
for sex_label in ["male", "female"]:
    mask = X_test["sex_male"] == (1 if sex_label == "male" else 0)
    if sex_label == "female":
        mask = X_test["sex_male"] == 0
    else:
        mask = X_test["sex_male"] == 1
    n = mask.sum()
    if n < 5:
        continue
    yt = y_test[mask]
    proba_sub = gbc.predict_proba(X_test_s[mask])[:, 1]
    try:
        auc_sub = roc_auc_score(yt, proba_sub)
    except ValueError:
        auc_sub = float("nan")
    subgroup_rows.append(dict(subgroup=sex_label, n=int(n), positive_rate=float(yt.mean()), auc=auc_sub))

subgroup_df = pd.DataFrame(subgroup_rows)
subgroup_df.to_csv("figures/subgroup_performance.csv", index=False)
print("\nSubgroup performance (Gradient Boosting):")
print(subgroup_df)

# ---------------------------------------------------------------------
# 6. Save headline numbers for the report
# ---------------------------------------------------------------------
summary = {
    "n_total": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "elevated_bp_prevalence": float(y.mean()),
    "bp_tertile_cutoff": float(bp_tertile_cutoff),
    "logreg_auc": results["Logistic Regression"]["auc"],
    "logreg_acc": results["Logistic Regression"]["acc"],
    "gbc_auc": results["Gradient Boosting"]["auc"],
    "gbc_acc": results["Gradient Boosting"]["acc"],
    "top_permutation_features": perm_df.sort_values(ascending=False).index[:5].tolist(),
    "top_logreg_features_positive": coef_df.sort_values(ascending=False).index[:3].tolist(),
    "top_logreg_features_negative": coef_df.sort_values().index[:3].tolist(),
}
import json
with open("figures/summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary:", summary)
