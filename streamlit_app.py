import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from streamlit_shap import st_shap

# Set page config
st.set_page_config(
    page_title="ChurnShield - Customer Churn Predictor",
    page_icon="🛡️",
    layout="wide",
)

# Load model and encoder
model = joblib.load("xgb_model.pkl")
encoder = joblib.load("label_encoders.pkl")

# Default values for resetting inputs
DEFAULT_VALUES = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1,
    "MonthlyCharges": 29.85
}

# Reset inputs
if "reset" not in st.session_state:
    st.session_state.reset = False

if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state.reset = True
else:
    st.session_state.reset = False

st.title("🛡️ ChurnShield - Customer Churn Predictor")
st.markdown("""
Welcome to **ChurnShield**, a machine learning-powered app designed to predict customer churn based on service usage and demographic information. 🔍📉

Use the sidebar to input customer details. Our model will predict whether the customer is likely to churn and explain which features influenced the decision.
""")

st.sidebar.header("📋 Customer Profile")

values = {}

def get_input(label, options=None, default=None, help_text=None):
    if options:
        return st.sidebar.selectbox(label, options, index=options.index(default), help=help_text)
    return st.sidebar.slider(label, 0.0, 150.0, float(default), help=help_text)

# Dropdown features
for key in [
    ("gender", ["Male", "Female"], "Customer's gender."),
    ("SeniorCitizen", ["No", "Yes"], "Is the customer a senior citizen?"),
    ("Partner", ["No", "Yes"], "Does the customer have a partner?"),
    ("Dependents", ["No", "Yes"], "Does the customer have dependents?"),
    ("PhoneService", ["No", "Yes"], "Is phone service active?"),
    ("MultipleLines", ["No", "Yes", "No phone service"], "Does the customer have multiple lines?"),
    ("InternetService", ["DSL", "Fiber optic", "No"], "Type of internet service."),
    ("OnlineSecurity", ["No", "Yes", "No internet service"], "Is online security enabled?"),
    ("OnlineBackup", ["No", "Yes", "No internet service"], "Is online backup enabled?"),
    ("DeviceProtection", ["No", "Yes", "No internet service"], "Is device protection enabled?"),
    ("TechSupport", ["No", "Yes", "No internet service"], "Is tech support active?"),
    ("StreamingTV", ["No", "Yes", "No internet service"], "Is streaming TV used?"),
    ("StreamingMovies", ["No", "Yes", "No internet service"], "Is streaming movies used?"),
    ("Contract", ["Month-to-month", "One year", "Two year"], "Type of contract."),
    ("PaperlessBilling", ["No", "Yes"], "Is billing paperless?"),
    ("PaymentMethod", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], "Customer's payment method."),
]:
    field, choices, tooltip = key
    values[field] = get_input(label=field.replace("_", " ").title(), options=choices, default=DEFAULT_VALUES[field], help_text=tooltip)

# Tenure
tenure = st.sidebar.slider(
    "🕓 Tenure (in months)", 0, 72, DEFAULT_VALUES["tenure"],
    help="How long the customer has been using the service."
)

# Average Monthly Charge input
avg_monthly_charge = st.sidebar.slider(
    "💸 Average Monthly Bill", 0.0, 150.0, DEFAULT_VALUES["MonthlyCharges"],
    help="Estimate of the customer's typical monthly charge."
)

# Automatically compute
monthly_charges = avg_monthly_charge
total_charges = avg_monthly_charge * tenure

# Add computed values
values["tenure"] = tenure
values["MonthlyCharges"] = monthly_charges
values["TotalCharges"] = total_charges

# Convert inputs to DataFrame
df = pd.DataFrame([values])

# Encode
for col in df.columns:
    if df[col].dtype == object:
        le = encoder.get(col)
        if le:
            df[col] = le.transform(df[col])

# Predict
pred_proba = model.predict_proba(df)[0][1]
pred = model.predict(df)[0]

st.subheader("🔍 Prediction Result")

col1, col2 = st.columns([1, 3])

with col1:
    st.metric(
        label="Churn Probability",
        value=f"{pred_proba*100:.2f}%",
        delta="High Risk" if pred_proba > 0.5 else "Low Risk",
        delta_color="inverse" if pred_proba > 0.5 else "normal"
    )

with col2:
    st.write(
        f"Based on the inputs provided, this customer is **{'likely' if pred else 'not likely'} to churn**."
    )

# SHAP values
st.subheader("📊 Feature Importance with SHAP")
explainer = shap.Explainer(model)
shap_values = explainer(df)

st_shap(shap.plots.waterfall(shap_values[0]), height=400)

with st.expander("📘 What is SHAP?"):
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** is a technique used to explain individual predictions. 

    In the above plot:
    - Each bar shows how a feature pushes the prediction toward **churn** (red) or **no churn** (blue).
    - The longer the bar, the more impact that feature has.

    This helps understand *why* the model made a certain decision.
    """)



