import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load model and scaler
model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# Default inputs
DEFAULT_INPUTS = {
    "Gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 800.0
}

# Set page title
st.set_page_config(page_title="ChurnShield", layout="wide")

# Title
st.title("🔐 ChurnShield – Telecom Churn Prediction & Explanation")
st.write("Predict whether a customer will churn and understand the reasons behind it using Explainable AI (SHAP).")

# Sidebar
st.sidebar.header("📝 Input Customer Details")

# Reset All Inputs
if st.sidebar.button("🔄 Reset All Inputs"):
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[key] = value
    st.rerun()

# Load current values
values = {key: st.session_state.get(key, DEFAULT_INPUTS[key]) for key in DEFAULT_INPUTS}

# Sidebar Inputs
gender = st.sidebar.selectbox("What is the customer's gender?", ["Male", "Female"], index=["Male", "Female"].index(values["Gender"]))
senior_citizen_label = st.sidebar.selectbox("Is the customer a senior citizen?", ["No", "Yes"], index=["No", "Yes"].index(values["SeniorCitizen"]))
senior_citizen = 1 if senior_citizen_label == "Yes" else 0
partner = st.sidebar.selectbox("Does the customer have a partner?", ["Yes", "No"], index=["Yes", "No"].index(values["Partner"]))
dependents = st.sidebar.selectbox("Does the customer have dependents?", ["Yes", "No"], index=["Yes", "No"].index(values["Dependents"]))
tenure = st.sidebar.slider("How many months has the customer stayed with the company?", 0, 72, int(values["tenure"]))
phone_service = st.sidebar.selectbox("Does the customer have a phone service?", ["Yes", "No"], index=["Yes", "No"].index(values["PhoneService"]))
multiple_lines = st.sidebar.selectbox("Does the customer have multiple lines?", ["Yes", "No", "No phone service"], index=["Yes", "No", "No phone service"].index(values["MultipleLines"]))
internet_service = st.sidebar.selectbox("What type of internet service does the customer use?", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(values["InternetService"]))
online_security = st.sidebar.selectbox("Does the customer have online security?", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(values["OnlineSecurity"]))
online_backup = st.sidebar.selectbox("Does the customer have online backup?", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(values["OnlineBackup"]))
device_protection = st.sidebar.selectbox("Is the customer's device protected?", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(values["DeviceProtection"]))
tech_support = st.sidebar.selectbox("Does the customer have tech support?", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(values["TechSupport"]))
streaming_tv = st.sidebar.selectbox("Does the customer stream TV?", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(values["StreamingTV"]))
streaming_movies = st.sidebar.selectbox("Does the customer stream movies?", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(values["StreamingMovies"]))
contract = st.sidebar.selectbox("What type of contract does the customer have?", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(values["Contract"]))
paperless_billing = st.sidebar.selectbox("Is billing paperless?", ["Yes", "No"], index=["Yes", "No"].index(values["PaperlessBilling"]))
payment_method = st.sidebar.selectbox("How does the customer pay?", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(values["PaymentMethod"]))
monthly_charges = st.sidebar.number_input("What is the customer's monthly charge?", min_value=0.0, max_value=200.0, value=float(values["MonthlyCharges"]))
total_charges = st.sidebar.number_input("What is the customer's total charge till now?", min_value=0.0, max_value=10000.0, value=float(values["TotalCharges"]))

# Prepare input dataframe
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Encoding
input_data_encoded = pd.get_dummies(input_data)
missing_cols = set(scaler.feature_names_in_) - set(input_data_encoded.columns)
for col in missing_cols:
    input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[scaler.feature_names_in_]

# Scale
input_scaled = scaler.transform(input_data_encoded)

# Predict
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1]

# Output
st.subheader("📊 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ The customer is **likely to churn**. (Probability: {proba:.2f})")
else:
    st.success(f"✅ The customer is **not likely to churn**. (Probability: {proba:.2f})")

# SHAP Explanation
st.subheader("🔍 Why this prediction?")
shap_values = explainer.shap_values(input_data_encoded)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, input_data_encoded, plot_type="bar", show=False)
st.pyplot(fig)

with st.expander("📌 See full SHAP explanation"):
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_data_encoded.iloc[0]), height=300)


