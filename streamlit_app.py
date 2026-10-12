import json

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="ChurnGuard", layout="wide")


@st.cache_resource
def load_all():
    cal = joblib.load("churn_model.joblib")            # calibrated model (probabilities)
    bundle = joblib.load("explainer_bundle.joblib")    # preprocessor + XGBoost (SHAP)
    explainer = shap.TreeExplainer(bundle["model"])
    threshold = json.load(open("threshold.json"))["threshold"]
    ref = pd.read_csv("data/telco_clean.csv")          # used for dropdown choices
    return cal["model"], cal["columns"], bundle["preprocess"], bundle["names"], explainer, threshold, ref


cal_model, COLUMNS, preprocess, names, explainer, THRESHOLD, ref = load_all()
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


st.title("ChurnGuard: Customer Churn Prediction")
st.caption(f"Predicts churn risk and shows the top 3 reasons. Offer threshold: {THRESHOLD:.2f}")

tab1, tab2 = st.tabs(["Single customer", "Upload CSV"])

with tab1:
    with st.form("single_customer"):
        cols = st.columns(3)
        values = {}
        for i, c in enumerate(COLUMNS):
            box = cols[i % 3]
            if c in num_cols:
                values[c] = box.number_input(c, value=float(ref[c].median()))
            else:
                options = sorted(ref[c].unique().tolist())
                default = options.index(ref[c].mode()[0])
                values[c] = box.selectbox(c, options, index=default)
        submitted = st.form_submit_button("Predict")

    if submitted:
        df = prepare(pd.DataFrame([values]))
        p = float(cal_model.predict_proba(df)[:, 1][0])
        reasons = top_reasons(df)[0]
        c1, c2 = st.columns(2)
        c1.metric("Churn probability", f"{p:.1%}")
        c2.metric("Risk level", risk_level(p))
        st.subheader("Top 3 reasons")
        for i, r in enumerate(reasons, 1):
            st.write(f"{i}. {r}")

with tab2:
    f = st.file_uploader("Upload customer CSV", type=["csv"])
    if f is not None:
        raw = pd.read_csv(f)
        df = prepare(raw)
        probs = cal_model.predict_proba(df)[:, 1]
        reasons = top_reasons(df)
        out = pd.DataFrame({
            "customerID": raw["customerID"] if "customerID" in raw.columns else range(len(raw)),
            "churn_probability": probs.round(3),
            "risk_level": [risk_level(p) for p in probs],
            "top_reasons": ["; ".join(r) for r in reasons],
        }).sort_values("churn_probability", ascending=False)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download results", out.to_csv(index=False), "churn_predictions.csv")