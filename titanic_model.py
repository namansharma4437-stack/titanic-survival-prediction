"""
Titanic Survival Prediction
----------------------------
Loads Titanic-Dataset.csv, cleans it, engineers features,
trains a Random Forest classifier, and reports performance.

Run:  python titanic_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)

# ----------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------
df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)
print(df.isnull().sum())

# ----------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------
# Title extracted from Name (Mr, Mrs, Miss, Master, Rare, etc.)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Family size features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Deck from Cabin (first letter); missing cabin -> 'U' (unknown)
df["Deck"] = df["Cabin"].str[0].fillna("U")

# Fill missing Age using median age per Title (more accurate than a single global median)
df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
df["Age"] = df["Age"].fillna(df["Age"].median())  # safety net

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Fare with median (relevant for test sets, harmless here)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Age & Fare bins (helps tree models find cleaner splits)
df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, 100],
                       labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"])
df["FareBin"] = pd.qcut(df["Fare"], 4, labels=["Low", "Mid", "High", "VeryHigh"])

# ----------------------------------------------------------------
# 3. Select features & encode categoricals
# ----------------------------------------------------------------
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title",
            "FamilySize", "IsAlone", "Deck", "AgeBin", "FareBin"]

X = df[features].copy()
y = df["Survived"]

X = pd.get_dummies(X, columns=["Sex", "Embarked", "Title", "Deck", "AgeBin", "FareBin"],
                    drop_first=True)

# ----------------------------------------------------------------
# 4. Train/test split
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------
# 5. Train models
# ----------------------------------------------------------------
# Baseline: Logistic Regression (scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)

# Main model: Random Forest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------------
# 6. Evaluate
# ----------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba=None):
    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    if y_proba is not None:
        print("ROC AUC :", round(roc_auc_score(y_true, y_proba), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Died", "Survived"]))

evaluate("Logistic Regression", y_test, log_preds)
evaluate("Random Forest", y_test, rf_preds, rf_proba)

# 5-fold cross-validation for a more robust estimate
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print("\nRandom Forest 5-fold CV accuracy: {:.4f} (+/- {:.4f})".format(
    cv_scores.mean(), cv_scores.std()))

# ----------------------------------------------------------------
# 7. Feature importance
# ----------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))
