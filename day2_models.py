import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Data load
df = pd.read_csv("data/telco_clean.csv")
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# 2. Grouping Columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Text columns
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 3. XGBoost imbalance ratio (stayed / churned)
ratio = (y == 0).sum() / (y == 1).sum()

# 4. Lists of Models
models = {
    "Baseline (dummy)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=ratio, eval_metric="logloss", random_state=42),
}

# 5. Stratified 5-fold cross-validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    rows.append({
        "Model": name,
        "PR-AUC": scores["test_pr_auc"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
    })
    print(f"Done: {name}")

results = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).round(3)
print("\n", results.to_string(index=False))
results.to_csv("reports/model_comparison.csv", index=False)