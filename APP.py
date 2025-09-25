import streamlit as st
import numpy as np
import joblib
import os
# --- Set base directory for model and files ---
BASE_DIR = os.path.dirname(__file__)
# --- Load model, scaler, and feature columns safely ---
try:
    model = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or related files: {e}")
    st.stop()
# --- Streamlit UI ---
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter the patient details below to predict the risk of heart disease.")
# --- Numeric inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=55)
resting_bp = st.number_input("Resting BP", min_value=50, max_value=250, value=130)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=250)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])
max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=140)
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
# --- Categorical inputs ---
sex = st.selectbox("Sex", ["F", "M"])
chest_pain = st.selectbox("Chest Pain Type", ["Other", "ATA", "NAP", "TA"])
resting_ecg = st.selectbox("Resting ECG", ["Other", "Normal", "ST"])
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
