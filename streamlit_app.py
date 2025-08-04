import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -----------------------
# Load Model & Columns
# -----------------------
MODEL_PATH = "churnshield_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    st.error(f"❌ Model file not found: {e}")
    st.stop()

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="ChurnShield", page_icon="🛡️", layout="centered")

# -----------------------
# App Title
# -----------------------
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>🛡️<span style='color:#F48FB1;'>ChurnShield – Customer Churn Prediction</span></h1>
        <p style='font-size:18px;'>Predict • Prevent • Retain</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Default Input Values
# -----------------------
default_inputs = {
    "What is your Gender?": "Female",
    "Are you a Senior Citizen?": "No",  
    "Do you share your subscription with a Partner?": "No",
    "Are there any dependents on your subscription?": "No",
    "What is the total Tenure of your subscription?": 0,
    "Is Phone Service available?": "No",
    "Can you use your subsciption on multiple devices?": "No",
    "Do you have Internet Access?": "No",
    "Do you have online Security?": "No",
    "Do you have Online Backup?": "No",
    "Do you have Device Protection?": "No",
    "Is Tech Support easily available?": "No",
    "TV Streaming?": "No",
    "Can you stream Movies?": "No",
    "Nature of Contract": "Month-to-month",
    "Is your Billing Paperless?": "No",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 0.0,
    "TotalCharges": 0.0,
}

# -----------------------
# Reset Logic
# -----------------------
if "reset" not in st.session_state:
    st.session_state.reset = False

def reset_inputs():
    st.session_state.reset = True


# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("📝 Input Customer Details")

# Reset button functionality
if st.sidebar.button("🔄 Reset All Inputs"):
    for key, value in default_inputs.items():
        st.session_state[key] = value
    st.rerun()

# Get current values from session state
values = {key: st.session_state.get(key, default_inputs[key]) for key in default_inputs}

# Sidebar widgets
gender = st.sidebar.selectbox(
    "What is your Gender?", ["Male", "Female"], 
    index=["Male", "Female"].index(values["Gender"])
)

senior_citizen_label = st.sidebar.selectbox(
    "Are you a Senior Citizen?", ["No", "Yes"],
    index=["No", "Yes"].index(values["SeniorCitizen"])
)
senior_citizen = 1 if senior_citizen_label == "Yes" else 0

partner = st.sidebar.selectbox(
    "Do you share your subscription with a Partner?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["Partner"])
)

dependents = st.sidebar.selectbox(
    "Are there any dependents on your subscription?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["Dependents"])
)

tenure = st.sidebar.slider(
    "What is the total Tenure of your subscription? (in months)", 0, 100, 
    int(values["tenure"])
)

phone_service = st.sidebar.selectbox(
    "Is Phone Service available?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["PhoneService"])
)

multiple_lines = st.sidebar.selectbox(
    "Can you use your subsciption on multiple devices?", ["Yes", "No", "No phone service"], 
    index=["Yes", "No", "No phone service"].index(values["MultipleLines"])
)

internet_service = st.sidebar.selectbox(
    "Do you have Internet Access?", ["DSL", "Fiber optic", "No"], 
    index=["DSL", "Fiber optic", "No"].index(values["InternetService"])
)

online_security = st.sidebar.selectbox(
    "Do you have online Security?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["OnlineSecurity"])
)

online_backup = st.sidebar.selectbox(
    "Do you have Online Backup?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["OnlineBackup"])
)

device_protection = st.sidebar.selectbox(
    "Do you have Device Protection?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["DeviceProtection"])
)

tech_support = st.sidebar.selectbox(
    "Is Tech Support easily available?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["TechSupport"])
)

streaming_tv = st.sidebar.selectbox(
    "TV Streaming?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["StreamingTV"])
)

streaming_movies = st.sidebar.selectbox(
    "Can you stream Movies?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["StreamingMovies"])
)

contract = st.sidebar.selectbox(
    "Nature of Contract", ["Month-to-month", "One year", "Two year"], 
    index=["Month-to-month", "One year", "Two year"].index(values["Contract"])
)

paperless_billing = st.sidebar.selectbox(
    "Is your Billing Paperless?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["PaperlessBilling"])
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(values["PaymentMethod"])
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges", min_value=0.0, max_value=200.0, value=float(values["MonthlyCharges"])
)

total_charges = st.sidebar.number_input(
    "Total Charges", min_value=0.0, max_value=10000.0, value=float(values["TotalCharges"])
)
# -----------------------
# Prepare Input Data
# -----------------------
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

st.write("### 📊 Customer Input Summary")
st.dataframe(input_data.T, use_container_width=True)

input_encoded = pd.get_dummies(input_data)
for col in model_columns:
    if col not in input_encoded:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# -----------------------
# Prediction & SHAP
# -----------------------
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1] * 100
    loyalty_score = 100 - prob

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=loyalty_score,
        title={'text': "Loyalty Score (0=Churn Likely, 100=Safe)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#F48FB1"},
               'steps': [
                   {'range': [0, 50], 'color': "#FFB6C1"},
                   {'range': [50, 80], 'color': "#FFDAB9"},
                   {'range': [80, 100], 'color': "#C1FFC1"}
               ]}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Prediction text
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to **Churn** (Probability: **{prob:.2f}%**).")
    else:
        st.success(f"✅ The customer is **Not likely to Churn** (Churn probability: **{prob:.2f}%**).")

    # SHAP Explanation
    st.write("### 🔍 Why this Prediction?")
    shap_values = explainer.shap_values(input_encoded)
    shap_df = pd.DataFrame({
        "Feature": input_encoded.columns,
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False).head(5)

    st.bar_chart(shap_df.set_index("Feature"))

# -----------------------
# Footer
# -----------------------
st.markdown(
    """
    ---
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Made with ❤️ using Streamlit | ChurnShield v1.2
    </div>
    """,
    unsafe_allow_html=True
)





