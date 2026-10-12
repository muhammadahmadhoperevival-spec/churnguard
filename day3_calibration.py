import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

df = pd.read_csv("data/telco_clean.csv")
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Class weights : Handling imbalance threshold and clean probabilities
raw = Pipeline([("prep", preprocess),
                ("model", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                                        eval_metric="logloss", random_state=42, n_jobs=-1))])
calibrated = CalibratedClassifierCV(raw, method="isotonic", cv=3)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_raw = cross_val_predict(raw, X, y, cv=cv, method="predict_proba")[:, 1]
p_cal = cross_val_predict(calibrated, X, y, cv=cv, method="predict_proba")[:, 1]

#
print(f"Brier raw:        {brier_score_loss(y, p_raw):.4f}")
print(f"Brier calibrated: {brier_score_loss(y, p_cal):.4f}")
print(f"PR-AUC raw:        {average_precision_score(y, p_raw):.3f}")
print(f"PR-AUC calibrated: {average_precision_score(y, p_cal):.3f}")


pd.DataFrame({"churn": y, "prob": p_cal}).to_csv("reports/oof_probs.csv", index=False)

# Final Model
calibrated.fit(X, y)
joblib.dump({"model": calibrated, "columns": X.columns.tolist()}, "churn_model.joblib")
print("Saved churn_model.joblib")