"""
AI RiskGuard - Streamlit App
Upload a dataset, pick a target column and sensitive features,
and get an automated bias/risk audit report with charts.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from risk_auditor import RiskAuditor

st.set_page_config(page_title="AI RiskGuard", page_icon="🛡️", layout="wide")

st.title("🛡️ AI RiskGuard — Model Bias & Risk Auditor")
st.write(
    "Upload a dataset to automatically check for missing values, class imbalance, "
    "bias across sensitive features, and (optionally) data drift against a reference set."
)

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
reference_file = st.file_uploader(
    "Optional: upload a reference/training dataset for drift detection (CSV)", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    target_col = st.selectbox("Select target/label column (optional)", ["None"] + columns)
    target_col = None if target_col == "None" else target_col

    sensitive_features = st.multiselect(
        "Select sensitive feature columns (e.g. gender, age_group, region)",
        [c for c in columns if c != target_col]
    )

    if st.button("Run Risk Audit"):
        auditor = RiskAuditor(df, sensitive_features=sensitive_features, target_col=target_col)

        reference_df = None
        if reference_file:
            reference_df = pd.read_csv(reference_file)

        report = auditor.run_full_audit(reference_df=reference_df)

        st.subheader("📊 Overall Risk Score")
        score = report["overall_risk_score"]
        level = report["overall_risk_level"]
        color = {"Low": "green", "Medium": "orange", "High": "red"}[level]
        st.markdown(f"### Score: {score}/100 — Risk Level: :{color}[{level}]")
        st.progress(score / 100)

        st.subheader("🧩 Missing Values")
        st.json(report["missing_values"])

        st.subheader("⚖️ Class Imbalance")
        st.json(report["class_imbalance"])
        if "distribution_pct" in report["class_imbalance"]:
            fig, ax = plt.subplots()
            dist = report["class_imbalance"]["distribution_pct"]
            ax.bar(dist.keys(), dist.values())
            ax.set_ylabel("Percentage")
            ax.set_title("Class Distribution")
            st.pyplot(fig)

        st.subheader("🧑‍🤝‍🧑 Sensitive Feature Bias")
        st.json(report["sensitive_feature_bias"])

        if "data_drift" in report:
            st.subheader("📈 Data Drift")
            st.json(report["data_drift"])

        st.success("Audit complete. Review flagged items above before deploying this model.")
else:
    st.info("Upload a CSV file to get started.")
