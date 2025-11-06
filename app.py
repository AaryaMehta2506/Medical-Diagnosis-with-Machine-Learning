import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
MODEL_PATH = 'medical_diagnosis_rf.joblib'
SCALER_PATH = 'medical_scaler.joblib'
LABELENC_PATH = 'medical_label_encoder.joblib'

# Load model and preprocessors
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABELENC_PATH)

# Define expected input columns
numeric_cols = ['Age', 'Billing Amount', 'Length_of_Stay_days']
cat_cols = ['Gender', 'Blood Type', 'Admission Type', 'Test Results']

st.title("Medical Diagnosis Prediction")
st.subheader("Predict the medical condition based on patient details")

# Sidebar inputs
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
blood_type = st.sidebar.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
billing_amount = st.sidebar.number_input("Billing Amount", min_value=0.0, value=5000.0)
admission_type = st.sidebar.selectbox("Admission Type", ["Emergency", "Routine", "Urgent", "Elective"])
test_results = st.sidebar.selectbox("Test Results", ["Normal", "Abnormal", "Inconclusive"])
length_of_stay = st.sidebar.number_input("Length of Stay (days)", min_value=0, value=3)

# Create a dataframe for input
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Blood Type': blood_type,
    'Billing Amount': billing_amount,
    'Admission Type': admission_type,
    'Test Results': test_results,
    'Length_of_Stay_days': length_of_stay
}])

# One-hot encode categorical variables
X_train_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
X_enc = pd.get_dummies(input_data, columns=cat_cols, drop_first=True)

# Add missing columns to match training
if X_train_cols is not None:
    for col in X_train_cols:
        if col not in X_enc.columns:
            X_enc[col] = 0
    X_enc = X_enc[X_train_cols]

# Scale numeric features together (fix)
X_enc[numeric_cols] = scaler.transform(X_enc[numeric_cols])

# Prediction
if st.button("Predict"):
    pred_idx = int(model.predict(X_enc)[0])
    pred_label = le.inverse_transform([pred_idx])[0]
    st.success(f"Predicted Medical Condition: {pred_label}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_enc)[0]
        prob_df = pd.DataFrame({
            'Condition': le.inverse_transform(np.arange(len(probs))),
            'Probability': probs
        }).sort_values(by='Probability', ascending=False).reset_index(drop=True)
        st.subheader("Prediction Confidence")
        st.dataframe(prob_df.head(5))
