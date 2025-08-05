import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and encoder
model = joblib.load("churnshield_model.pkl")
encoder = joblib.load("model_columns.pkl")

# Define the input features required by the model (except calculated ones)
INPUT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# Define user-friendly prompts
PROMPTS = {
    "gender": "What is your gender?",
    "SeniorCitizen": "Are you a senior citizen?",
    "Partner": "Do you have a partner?",
    "Dependents": "Do you have any dependents?",
    "tenure": "How many months have you been a customer?",
    "PhoneService": "Do you use phone service?",
    "MultipleLines": "Do you have multiple lines?",
    "InternetService": "What kind of internet service do you use?",
    "OnlineSecurity": "Do you have online security?",
    "OnlineBackup": "Do you use online backup?",
    "DeviceProtection": "Do you have device protection?",
    "TechSupport": "Do you use tech support?",
    "StreamingTV": "Do you stream TV?",
    "StreamingMovies": "Do you stream movies?",
    "Contract": "What is the contract type?",
    "PaperlessBilling": "Do you use paperless billing?",
    "PaymentMethod": "What is your payment method?"
}

# Default values
DEFAULT_VALUES = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

# Streamlit UI
st.set_page_config(page_title="ChurnShield: Customer Churn Predictor", layout="wide")
st.title("🔐 ChurnShield")
st.subheader("Smart, Explainable Churn Prediction for Telecom Users")

# Sidebar
st.sidebar.header("Enter Customer Details")

# Reset button
if st.sidebar.button("Reset All Inputs"):
    st.experimental_rerun()

user_inputs = {}

for col in INPUT_FEATURES:
    if col == "tenure":
        user_inputs[col] = st.sidebar.slider(PROMPTS[col], 0, 72, DEFAULT_VALUES[col])
    elif col in ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        user_inputs[col] = st.sidebar.radio(PROMPTS[col], ["Yes", "No"], index=["Yes", "No"].index(DEFAULT_VALUES[col]))
    else:
        options = sorted(encoder[col].classes_.tolist())
        default_idx = options.index(DEFAULT_VALUES[col]) if DEFAULT_VALUES[col] in options else 0
        user_inputs[col] = st.sidebar.selectbox(PROMPTS[col], options, index=default_idx)

# Auto-calculated values
monthly_charge = 20 + user_inputs["tenure"] * 0.5  # simplistic assumption
total_charge = monthly_charge * user_inputs["tenure"]
st.sidebar.markdown(f"💰 **Monthly Charges:** ${monthly_charge:.2f}")
st.sidebar.markdown(f"💰 **Total Charges:** ${total_charge:.2f}")

# Add to inputs
user_inputs["MonthlyCharges"] = monthly_charge
user_inputs["TotalCharges"] = total_charge

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs])

# Encode categorical variables
for col in input_df.columns:
    if col in encoder:
        le = encoder[col]
        input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.markdown(f"**Churn Prediction:** {'❌ Will Churn' if prediction else '✅ Will Not Churn'}")
    st.markdown(f"**Confidence:** {prob*100:.2f}%")

    # SHAP explanation
    st.subheader("🔍 Why this prediction?")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    st.markdown("The plot below shows which features most influenced the model’s decision.")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
