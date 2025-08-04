import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from streamlit_shap import st_shap

# Set page config
st.set_page_config(
    page_title="ChurnShield - Customer Churn Predictor",
    page_icon="🛡️",
    layout="wide",
)

# Load model and encoder
model = joblib.load("churnshield_model.pkl")
encoder = joblib.load("model_columns.pkl")

# Default input values (single source of truth)
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

# Reset flag
if "reset" not in st.session_state:
    st.session_state.reset = False

if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state.reset = True

st.title("🛡️ ChurnShield - Customer Churn Predictor")
st.markdown("""
Welcome to **ChurnShield**, a machine learning-powered app designed to predict customer churn based on service usage and demographic information. 🔍📉

Use the sidebar to input customer details. Our model will predict whether the customer is likely to churn and explain which features influenced the decision.
""")

st.sidebar.header("📋 Customer Profile")

values = {}

def get_input(label, options=None, default=None):
    if options:
        return st.sidebar.selectbox(label, options, index=options.index(default))
    return st.sidebar.slider(label, 0.0, 150.0, float(default))

# All fields and user-friendly labels
fields = [
    ("gender", ["Male", "Female"], "What is your Gender?"),
    ("SeniorCitizen", ["No", "Yes"], "Is the Customer a Senior Citizen?"),
    ("Partner", ["No", "Yes"], "Do they have a Partner?"),
    ("Dependents", ["No", "Yes"], "Do they have Dependents?"),
    ("PhoneService", ["No", "Yes"], "Is Phone Service Active?"),
    ("MultipleLines", ["No", "Yes", "No phone service"], "Do they have Multiple Lines?"),
    ("InternetService", ["DSL", "Fiber optic", "No"], "What type of Internet Service?"),
    ("OnlineSecurity", ["No", "Yes", "No internet service"], "Is Online Security enabled?"),
    ("OnlineBackup", ["No", "Yes", "No internet service"], "Is Online Backup enabled?"),
    ("DeviceProtection", ["No", "Yes", "No internet service"], "Is Device Protection active?"),
    ("TechSupport", ["No", "Yes", "No internet service"], "Do they have Tech Support?"),
    ("StreamingTV", ["No", "Yes", "No internet service"], "Do they use Streaming TV?"),
    ("StreamingMovies", ["No", "Yes", "No internet service"], "Do they use Streaming Movies?"),
    ("Contract", ["Month-to-month", "One year", "Two year"], "What type of Contract?"),
    ("PaperlessBilling", ["No", "Yes"], "Is Paperless Billing enabled?"),
    ("PaymentMethod", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], "What is the Payment Method?")
]

# Sidebar form
for key, choices, question in fields:
    default_val = DEFAULT_VALUES[key] if not st.session_state.reset else DEFAULT_VALUES[key]
    values[key] = get_input(label=question, options=choices, default=default_val)

# Tenure
tenure = st.sidebar.slider(
    "🕓 How many months has the customer been active?", 0, 72, DEFAULT_VALUES["tenure"] if not st.session_state.reset else DEFAULT_VALUES["tenure"]
)

# Average Monthly Charge input
avg_monthly_charge = st.sidebar.slider(
    "💸 Average Monthly Bill", 0.0, 150.0, DEFAULT_VALUES["MonthlyCharges"] if not st.session_state.reset else DEFAULT_VALUES["MonthlyCharges"]
)

# Auto compute
monthly_charges = avg_monthly_charge
total_charges = avg_monthly_charge * tenure

# Add computed
values["tenure"] = tenure
values["MonthlyCharges"] = monthly_charges
values["TotalCharges"] = total_charges

# Reset flag off after input
st.session_state.reset = False

# DataFrame
df = pd.DataFrame([values])

# Encoding
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

# SHAP
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
