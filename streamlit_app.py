import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# Set page config
st.set_page_config(
    page_title="ChurnShield - Customer Churn Predictor",
    page_icon="🛡️",
    layout="wide",
)

# Load model and encoder
model = joblib.load("churnshield_model.pkl")
encoder = joblib.load("model_columns.pkl")

# Default values
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

if "inputs" not in st.session_state or st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state.inputs = DEFAULT_VALUES.copy()

st.title("🛡️ ChurnShield - Customer Churn Predictor")
st.markdown("""
Welcome to **ChurnShield**, a machine learning-powered app that predicts customer churn based on service usage and profile. 🔍📉

Use the sidebar to enter customer details and get predictions with interpretability.
""")

st.sidebar.header("📋 Customer Profile")
inputs = {}

def ask_question(key, question, options=None, default=None, help_text=None):
    default_val = st.session_state.inputs.get(key, default)
    if options:
        inputs[key] = st.sidebar.selectbox(question, options, index=options.index(default_val), help=help_text)
    else:
        inputs[key] = st.sidebar.slider(question, 0.0, 150.0, float(default_val), help=help_text)

questions = [
    ("gender", "What is your Gender?", ["Male", "Female"]),
    ("SeniorCitizen", "Are you a senior citizen?", ["No", "Yes"]),
    ("Partner", "Do you have a partner?", ["No", "Yes"]),
    ("Dependents", "Do you have any dependents?", ["No", "Yes"]),
    ("PhoneService", "Is phone service active?", ["No", "Yes"]),
    ("MultipleLines", "Do you have multiple lines?", ["No", "Yes", "No phone service"]),
    ("InternetService", "Which internet service do you use?", ["DSL", "Fiber optic", "No"]),
    ("OnlineSecurity", "Is online security enabled?", ["No", "Yes", "No internet service"]),
    ("OnlineBackup", "Is online backup enabled?", ["No", "Yes", "No internet service"]),
    ("DeviceProtection", "Is device protection enabled?", ["No", "Yes", "No internet service"]),
    ("TechSupport", "Do you have tech support?", ["No", "Yes", "No internet service"]),
    ("StreamingTV", "Do you use streaming TV?", ["No", "Yes", "No internet service"]),
    ("StreamingMovies", "Do you use streaming movies?", ["No", "Yes", "No internet service"]),
    ("Contract", "What is your contract type?", ["Month-to-month", "One year", "Two year"]),
    ("PaperlessBilling", "Is paperless billing enabled?", ["No", "Yes"]),
    ("PaymentMethod", "Choose your payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
]

# Ask all dropdowns
for key, question, options in questions:
    ask_question(key, question, options=options, default=DEFAULT_VALUES[key])

# Tenure
inputs["tenure"] = st.sidebar.slider(
    "🕓 How many months have you used the service?", 0, 72, DEFAULT_VALUES["tenure"],
    help="This represents the duration (in months) of service usage."
)

# Average Monthly Charges
inputs["MonthlyCharges"] = st.sidebar.slider(
    "💸 What is your average monthly bill?", 0.0, 150.0, DEFAULT_VALUES["MonthlyCharges"],
    help="An estimate of the average monthly charge."
)

# Total Charges is calculated automatically
inputs["TotalCharges"] = inputs["MonthlyCharges"] * inputs["tenure"]

# Save latest inputs
st.session_state.inputs = inputs.copy()

# Convert to DataFrame
df = pd.DataFrame([inputs])

# Encode categorical values
for col in df.columns:
    if df[col].dtype == object:
        le = encoder.get(col)
        if le:
            df[col] = le.transform(df[col])

# Predict
pred_proba = model.predict_proba(df)[0][1]
pred = model.predict(df)[0]

# Result section
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
        f"Based on the data, this customer is **{'likely' if pred else 'not likely'} to churn**."
    )

# SHAP Plot
st.subheader("📊 Feature Importance with SHAP")
explainer = shap.Explainer(model)
shap_values = explainer(df)
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(bbox_inches='tight')

with st.expander("📘 What is SHAP?"):
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** explains individual predictions.

    In the plot above:
    - **Red** bars push toward churn.
    - **Blue** bars push toward staying.
    - The longer the bar, the more it influences the decision.

    This helps us understand *why* the model made a decision.
    """)

