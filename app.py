import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Prostate Cancer Predictor",
    page_icon="🩺",
    layout="centered"
)

# --- Load Model & Scaler ---
model  = joblib.load('prostate_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Header ---
st.title("🩺 Prostate Cancer Prediction")
st.markdown("Enter the patient's diagnostic measurements below to predict whether the tumor is **Malignant** or **Benign**.")
st.divider()

# --- Input Form ---
st.subheader("📋 Patient Measurements")

col1, col2 = st.columns(2)

with col1:
    radius      = st.number_input("Radius",      min_value=0.0, value=18.0, step=0.1)
    perimeter   = st.number_input("Perimeter",   min_value=0.0, value=110.0, step=0.1)
    smoothness  = st.number_input("Smoothness",  min_value=0.0, value=0.105, step=0.001, format="%.3f")

with col2:
    compactness = st.number_input("Compactness", min_value=0.0, value=0.15,  step=0.001, format="%.3f")
    symmetry    = st.number_input("Symmetry",    min_value=0.0, value=0.20,  step=0.001, format="%.3f")

st.divider()

# --- Predict Button ---
if st.button("🔍 Predict", use_container_width=True):

    input_data = pd.DataFrame([{
        'radius':      radius,
        'perimeter':   perimeter,
        'smoothness':  smoothness,
        'compactness': compactness,
        'symmetry':    symmetry
    }])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧾 Result")

    if prediction == 1:
        st.error(f"⚠️ **Malignant** — High risk of prostate cancer detected.")
    else:
        st.success(f"✅ **Benign** — No malignancy detected.")

    st.metric(label="Probability of Malignancy", value=f"{probability:.2%}")

    # Confidence bar
    st.progress(float(probability))
    st.caption("Bar represents malignancy probability (0% → 100%)")

# --- Footer ---
st.divider()
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")