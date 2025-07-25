'''import streamlit as st
import pandas as pd
import joblib
import os

# ===========================
# DEBUGGING INFO (REMOVE LATER)
# ===========================
st.write("Current Working Directory:", os.getcwd())
st.write("Files in this folder:", os.listdir())

# ===========================
# MODEL & COLUMNS LOADING
# ===========================
MODEL_PATH = os.path.join(os.getcwd(), "churnshield_model.pkl")
COLUMNS_PATH = os.path.join(os.getcwd(), "model_columns.pkl")

try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    st.success("✅ Model & columns loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# ===========================
# APP TITLE
# ===========================
st.title("💖 ChurnShield – Customer Churn Prediction")

st.markdown("""
Welcome to **ChurnShield**!  
Enter customer details below and let’s predict whether the customer will churn or stay loyal.
""")

# ===========================
# USER INPUTS
# ===========================
def user_input():
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    return pd.DataFrame([data])

input_df = user_input()

# ===========================
# DATA PREPROCESSING
# ===========================
df = pd.get_dummies(input_df)
for col in model_columns:
    if col not in df.columns:
        df[col] = 0
df = df[model_columns]

# ===========================
# PREDICTION
# ===========================
if st.button("Predict Churn"):
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn! (Probability: {probability:.2f}%)")
    else:
        st.success(f"💖 Customer is likely to stay. (Churn Probability: {probability:.2f}%)")

        '''
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
    "Gender": "Female",
    "SeniorCitizen": "No",  
    "Partner": "No",
    "Dependents": "No",
    "tenure": 0,
    "PhoneService": "No",
    "MultipleLines": "No",
    "InternetService": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "No",
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
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value
    st.rerun()

# Get current values from session state
values = {key: st.session_state.get(key, default_inputs[key]) for key in default_inputs}

# Sidebar widgets
gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"], 
    index=["Male", "Female"].index(values["Gender"])
)

senior_citizen_label = st.sidebar.selectbox(
    "Senior Citizen", ["No", "Yes"],
    index=["No", "Yes"].index(values["SeniorCitizen"])
)
senior_citizen = 1 if senior_citizen_label == "Yes" else 0

partner = st.sidebar.selectbox(
    "Partner", ["Yes", "No"], 
    index=["Yes", "No"].index(values["Partner"])
)

dependents = st.sidebar.selectbox(
    "Dependents", ["Yes", "No"], 
    index=["Yes", "No"].index(values["Dependents"])
)

tenure = st.sidebar.slider(
    "Tenure (in months)", 0, 72, 
    int(values["tenure"])
)

phone_service = st.sidebar.selectbox(
    "Phone Service", ["Yes", "No"], 
    index=["Yes", "No"].index(values["PhoneService"])
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"], 
    index=["Yes", "No", "No phone service"].index(values["MultipleLines"])
)

internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"], 
    index=["DSL", "Fiber optic", "No"].index(values["InternetService"])
)

online_security = st.sidebar.selectbox(
    "Online Security", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["OnlineSecurity"])
)

online_backup = st.sidebar.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["OnlineBackup"])
)

device_protection = st.sidebar.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["DeviceProtection"])
)

tech_support = st.sidebar.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["TechSupport"])
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["StreamingTV"])
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["StreamingMovies"])
)

contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"], 
    index=["Month-to-month", "One year", "Two year"].index(values["Contract"])
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing", ["Yes", "No"], 
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


