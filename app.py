import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Dropout Risk App", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")

model = load_model()

# --- SESSION STATE ---
if "results" not in st.session_state:
    st.session_state["results"] = None

st.title("📊 Student Dropout Risk Predictor")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

# --- CSV LOAD (cached) ---
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# --- CHECK IF FILE UPLOADED ---
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())
    X = df.drop(columns=["Class"], errors="ignore")

    # --- PREDICTION BUTTON ---
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

# --- ELSE: no file uploaded ---
else:
    st.info("📁 Please upload a CSV file to see predictions and risk scores.")
    st.write("Make sure your CSV contains the same features as used in training the model.")

# --- SHOW RESULTS ---
if st.session_state["results"] is not None:
    results = st.session_state["results"]
    st.subheader("🔥 Top 20 High-Risk Students")
    st.dataframe(results.sort_values("risk_score", ascending=False).head(20))
