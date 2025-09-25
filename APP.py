import streamlit as st
import numpy as np
import joblib
import os

# --- Load model, scaler and feature columns ---
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# --- Streamlit UI ---
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter the patient details below to predict the risk of heart disease.")

# Numeric inputs
age = st.number_input("Age", min_value=1, max_value=120, value=55)
resting_bp = st.number_input("Resting BP", min_value=50, max_value=250, value=130)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=250)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])
max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=140)
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)

# Categorical inputs
sex = st.selectbox("Sex", ["F", "M"])
chest_pain = st.selectbox("Chest Pain Type", ["Other", "ATA", "NAP", "TA"])
resting_ecg = st.selectbox("Resting ECG", ["Other", "Normal", "ST"])
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
st_slope = st.selectbox("ST Slope", ["Other", "Flat", "Up"])

# --- Build feature vector (must match training order) ---
def build_features():
    features = []
    features.append(age)
    features.append(resting_bp)
    features.append(cholesterol)
    features.append(fasting_bs)
    features.append(max_hr)
    features.append(oldpeak)

    features.append(1 if sex == "M" else 0)
    features.append(1 if chest_pain == "ATA" else 0)
    features.append(1 if chest_pain == "NAP" else 0)
    features.append(1 if chest_pain == "TA" else 0)
    features.append(1 if resting_ecg == "Normal" else 0)
    features.append(1 if resting_ecg == "ST" else 0)
    features.append(1 if exercise_angina == "Y" else 0)
    features.append(1 if st_slope == "Flat" else 0)
    features.append(1 if st_slope == "Up" else 0)

    X = np.array([features])
    # scale inputs
    X_scaled = scaler.transform(X)
    return X_scaled

# --- Prediction ---
if st.button("Predict"):
    X = build_features()
    pred = model.predict(X)[0]
    try:
        prob = model.predict_proba(X)[0][1]
    except Exception:
        prob = None

    if pred == 1:
        st.error("🚨 High risk of Heart Disease detected.")
    else:
        st.success("✅ Low risk of Heart Disease.")

    if prob is not None:
        st.write(f"**Probability of Heart Disease:** {prob*100:.1f}%")
