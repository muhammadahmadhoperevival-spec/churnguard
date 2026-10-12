import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

Xt = preprocess.fit_transform(X)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
names = [n.split("__", 1)[1] for n in preprocess.get_feature_names_out()]

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                      eval_metric="logloss", random_state=42, n_jobs=-1).fit(Xt, y)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xt)
Xt_df = pd.DataFrame(Xt, columns=names)

# Global explanation
shap.summary_plot(sv, Xt_df, show=False, max_display=15)
plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Local explanation:
def top_reasons(i, k=3):
    idx = np.argsort(-np.abs(sv[i]))[:k]
    return [(names[j], "raises risk" if sv[i][j] > 0 else "lowers risk") for j in idx]

risky = int(np.argmax(model.predict_proba(Xt)[:, 1]))
print("Highest-risk customer:", df.loc[risky, "customerID"])
for feat, effect in top_reasons(risky):
    print(f"  - {feat}: {effect}")

joblib.dump({"preprocess": preprocess, "model": model, "names": names,
             "columns": X.columns.tolist()}, "explainer_bundle.joblib")
print("Saved explainer_bundle.joblib")