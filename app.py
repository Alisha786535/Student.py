import streamlit as st
st.write("✅ Streamlit context OK")

import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="Student Dropout Risk App",
    layout="wide"
)

# =========================
# CACHE MODEL (VERY IMPORTANT)
# =========================
@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")

model = load_model()

# =========================
# APP TITLE
# =========================
st.title("📊 Student Dropout Risk Predictor")
st.write("Upload CSV to check student dropout risk")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Student CSV File",
    type=["csv"]
)

# =========================
# SESSION STATE INIT
# =========================
if "results" not in st.session_state:
    st.session_state["results"] = None

# =========================
# AFTER FILE UPLOAD
# =========================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    X = df.drop(columns=["Class"], errors="ignore")

    # =========================
    # RUN PREDICTION BUTTON
    # =========================
    if st.button("🚀 Run Prediction"):
        try:
            risk_scores = model.predict_proba(X)[:, 1]

            def risk_label(score):
                if score >= 0.7:
                    return "High"
                elif score >= 0.4:
                    return "Medium"
                else:
                    return "Low"

            results = pd.DataFrame({
                "student_id": X.index,
                "risk_score": risk_scores,
                "risk_label": [risk_label(s) for s in risk_scores]
            })

            st.session_state["results"] = results
            st.success("Prediction completed successfully ✅")

        except Exception as e:
            st.error("❌ CSV format training data se match nahi kar raha")
            st.exception(e)
            st.stop()

# =========================
# SHOW RESULTS SAFELY
# =========================
if st.session_state["results"] is not None:
    results = st.session_state["results"]

    # ---- TOP 20 HIGH RISK ----
    st.subheader("🔥 Top 20 High-Risk Students")
    top_20 = results.sort_values(
        by="risk_score",
        ascending=False
    ).head(20)
    st.dataframe(top_20)

    # ---- SINGLE STUDENT CHECK ----
    st.subheader("🎯 Check Individual Student Risk")
    idx = st.number_input(
        "Enter Student Index",
        min_value=0,
        max_value=len(results) - 1,
        value=0
    )

    student = results.iloc[idx]
    st.write(f"**Student ID:** {student['student_id']}")
    st.write(f"**Risk Score:** {student['risk_score']:.2f}")
    st.write(f"**Risk Level:** {student['risk_label']}")

    # ---- FEATURE IMPORTANCE ----
    st.subheader("📌 Top Reasons (Feature Importance)")
    try:
        feature_names = model.named_steps["preprocessing"].get_feature_names_out()
        coefs = model.named_steps["classifier"].coef_[0]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": coefs
        }).sort_values(
            by="Importance",
            key=abs,
            ascending=False
        ).head(10)

        st.table(importance_df)
    except:
        st.info("Feature importance available nahi hai")

    # ---- EXPORT ----
    if st.button("💾 Export Predictions"):
        results.to_csv("predictions.csv", index=False)
        st.success("predictions.csv saved successfully ✅")
