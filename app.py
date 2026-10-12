import json
import joblib
import numpy as np
import pandas as pd
import shap
import gradio as gr

# Models load karo
cal = joblib.load("churn_model.joblib")          # calibrated model (probability ke liye)
cal_model, COLUMNS = cal["model"], cal["columns"]
bundle = joblib.load("explainer_bundle.joblib")  # SHAP ke liye
preprocess, xgb_model, names = bundle["preprocess"], bundle["model"], bundle["names"]
explainer = shap.TreeExplainer(xgb_model)
THRESHOLD = json.load(open("threshold.json"))["threshold"]

ref = pd.read_csv("data/telco_clean.csv")        # dropdown choices ke liye
num_cols = ref[COLUMNS].select_dtypes(include="number").columns.tolist()

def prepare(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df[COLUMNS]

def risk_level(p):
    if p >= 0.5:
        return "HIGH"
    if p >= THRESHOLD:
        return "MEDIUM (send offer)"
    return "LOW"

def pretty(name):
    if "_" in name:
        col, val = name.split("_", 1)
        return f"{col} = {val}"
    return name

def top_reasons(df, k=3):
    Xt = preprocess.transform(df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    sv = explainer.shap_values(Xt)
    out = []
    for row in sv:
        idx = np.argsort(-np.abs(row))[:k]
        out.append([f"{pretty(names[j])} ({'increases' if row[j] > 0 else 'reduces'} risk)"
                    for j in idx])
    return out

def predict_single(*values):
    df = prepare(pd.DataFrame([dict(zip(COLUMNS, values))]))
    p = float(cal_model.predict_proba(df)[:, 1][0])
    reasons = top_reasons(df)[0]
    return f"{p:.1%}", risk_level(p), "\n".join(f"{i+1}. {r}" for i, r in enumerate(reasons))

def predict_csv(file):
    path = file if isinstance(file, str) else file.name
    raw = pd.read_csv(path)
    df = prepare(raw)
    probs = cal_model.predict_proba(df)[:, 1]
    reasons = top_reasons(df)
    out = pd.DataFrame({
        "customerID": raw["customerID"] if "customerID" in raw.columns else range(len(raw)),
        "churn_probability": probs.round(3),
        "risk_level": [risk_level(p) for p in probs],
        "top_reasons": ["; ".join(r) for r in reasons],
    })
    return out.sort_values("churn_probability", ascending=False)

with gr.Blocks(title="ChurnGuard") as demo:
    gr.Markdown("# ChurnGuard: Customer Churn Prediction\n"
                f"Predicts churn risk and shows the top 3 reasons. Offer threshold: **{THRESHOLD:.2f}**")
    with gr.Tab("Single customer"):
        inputs = []
        for c in COLUMNS:
            if c in num_cols:
                inputs.append(gr.Number(label=c, value=float(ref[c].median())))
            else:
                inputs.append(gr.Dropdown(sorted(ref[c].unique().tolist()),
                                          value=ref[c].mode()[0], label=c))
        btn = gr.Button("Predict")
        o1 = gr.Textbox(label="Churn probability")
        o2 = gr.Textbox(label="Risk level")
        o3 = gr.Textbox(label="Top 3 reasons", lines=3)
        btn.click(predict_single, inputs, [o1, o2, o3])
    with gr.Tab("Upload CSV"):
        f = gr.File(label="Upload customer CSV", file_types=[".csv"])
        b2 = gr.Button("Predict all")
        table = gr.Dataframe(label="Results (highest risk first)")
        b2.click(predict_csv, f, table)

demo.launch()