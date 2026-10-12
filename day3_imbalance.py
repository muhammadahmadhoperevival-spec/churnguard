import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline      # sklearn wali nahi, imblearn wali
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/telco_clean.csv")
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

ratio = (y == 0).sum() / (y == 1).sum()

def make_pipe(model, smote=False):
    steps = [("prep", preprocess)]
    if smote:
        # SMOTE Pipeline ke andar hai, isliye sirf training folds par lagta hai
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", model))
    return Pipeline(steps)

def lr(**kw):  return LogisticRegression(max_iter=1000, **kw)
def xgb(**kw): return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                                    eval_metric="logloss", random_state=42, n_jobs=-1, **kw)

experiments = {
    "LogReg - nothing":       make_pipe(lr()),
    "LogReg - class weights": make_pipe(lr(class_weight="balanced")),
    "LogReg - SMOTE":         make_pipe(lr(), smote=True),
    "XGB - nothing":          make_pipe(xgb()),
    "XGB - class weights":    make_pipe(xgb(scale_pos_weight=ratio)),
    "XGB - SMOTE":            make_pipe(xgb(), smote=True),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"pr_auc": "average_precision", "precision": "precision",
           "recall": "recall", "f1": "f1"}

rows = []
for name, pipe in experiments.items():
    s = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    rows.append({"Setup": name, "PR-AUC": s["test_pr_auc"].mean(),
                 "Precision": s["test_precision"].mean(),
                 "Recall": s["test_recall"].mean(), "F1": s["test_f1"].mean()})
    print("Done:", name)

res = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).round(3)
print("\n", res.to_string(index=False))
res.to_csv("reports/imbalance_comparison.csv", index=False)