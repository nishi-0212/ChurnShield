import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid", palette="pastel")
blue = "#90CAF9"
pink = "#F48FB1"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChurnShield", page_icon="🛡️", layout="wide")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    saved     = joblib.load("notebooks/models/churnshield_model.pkl")
    model     = saved["model"]
    threshold = saved["threshold"]
    columns   = joblib.load("notebooks/models/model_columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, columns, explainer, threshold

try:
    model, model_columns, explainer, threshold = load_model()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {text-align:center; padding: 1rem 0 0.5rem;}
    .main-header h1 {font-size: 2.4rem; font-weight: 800;}
    .main-header p  {font-size: 1.1rem; color: #6c757d;}
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border: 1px solid #dee2e6;
    }
    .churn-alert {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 1rem; border-radius: 6px; margin: 1rem 0;
    }
    .safe-alert {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 1rem; border-radius: 6px; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ <span style="color:#F48FB1;">ChurnShield</span></h1>
    <p>ML-powered customer churn prediction with SHAP explainability</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar — Customer Inputs ─────────────────────────────────────────────────
st.sidebar.header("📝 Customer Details")

if st.sidebar.button("🔄 Reset to Defaults"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Helper to get session value with default
def sv(key, default):
    return st.session_state.get(key, default)

# Demographics
st.sidebar.subheader("Demographics")
gender           = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen   = st.sidebar.selectbox("Senior Citizen?", ["No", "Yes"])
partner          = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents       = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
tenure           = st.sidebar.slider("Tenure (months)", 0, 72, 12)

# Services
st.sidebar.subheader("Services")
phone_service    = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

# FIX: consistent option values for internet service
internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No Internet Service"]
)

no_internet = internet_service == "No Internet Service"
inet_options = ["Yes", "No"] if not no_internet else ["No internet service"]

online_security  = st.sidebar.selectbox("Online Security",  inet_options)
online_backup    = st.sidebar.selectbox("Online Backup",    inet_options)
device_protection= st.sidebar.selectbox("Device Protection",inet_options)
tech_support     = st.sidebar.selectbox("Tech Support",     inet_options)
streaming_tv     = st.sidebar.selectbox("Streaming TV",     inet_options)
streaming_movies = st.sidebar.selectbox("Streaming Movies", inet_options)

# Billing
st.sidebar.subheader("Billing")
contract         = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless        = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method   = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges  = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=1.0)
total_charges    = round(monthly_charges * tenure, 2)
st.sidebar.markdown(f"**Auto-calculated Total Charges:** ${total_charges:,.2f}")

# ── Build input dataframe ─────────────────────────────────────────────────────
senior_int = 1 if senior_citizen == "Yes" else 0

input_data = pd.DataFrame([{
    "gender":            gender,
    "SeniorCitizen":     senior_int,
    "Partner":           partner,
    "Dependents":        dependents,
    "tenure":            tenure,
    "PhoneService":      phone_service,
    "MultipleLines":     multiple_lines,
    "InternetService":   internet_service,
    "OnlineSecurity":    online_security,
    "OnlineBackup":      online_backup,
    "DeviceProtection":  device_protection,
    "TechSupport":       tech_support,
    "StreamingTV":       streaming_tv,
    "StreamingMovies":   streaming_movies,
    "Contract":          contract,
    "PaperlessBilling":  paperless,
    "PaymentMethod":     payment_method,
    "MonthlyCharges":    monthly_charges,
    "TotalCharges":      total_charges,
    # Original engineered features
    "AvgChargesPerMonth":        round(total_charges / (tenure + 1), 2),
    "HighMonthlyCharge":         int(monthly_charges > 64.76),
    "IsLongTerm":                int(contract != "Month-to-month"),
    "Tenure_Charge_Interaction": tenure * monthly_charges,
    # New engineered features
    "ChargeGap":         monthly_charges - round(total_charges / (tenure + 1), 2),
    "NumServices":       sum([
                             phone_service    == "Yes",
                             online_security  == "Yes",
                             online_backup    == "Yes",
                             device_protection== "Yes",
                             tech_support     == "Yes",
                             streaming_tv     == "Yes",
                             streaming_movies == "Yes",
                         ]),
    "RevenueAtRisk":     round(monthly_charges * (1 - tenure / 73), 2),
    "IsNewCustomer":     int(tenure <= 6),
    "NoSupportServices": int(
                             online_security != "Yes" and
                             tech_support    != "Yes" and
                             online_backup   != "Yes"
                         ),
    "HighRiskPayment":   int(paperless == "Yes" and payment_method == "Electronic check"),
    "LogTenure":         round(np.log1p(tenure), 4),
}])

# ── Main panel ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📊 Customer Summary")
    display_df = input_data.T.copy()
    display_df.columns = ["Value"]
    st.dataframe(display_df, use_container_width=True, height=420)

with col_right:
    st.subheader("🔮 Prediction")
    predict_btn = st.button("Run Churn Prediction", type="primary", use_container_width=True)

    if predict_btn:
        # Encode
        input_encoded = pd.get_dummies(input_data)
        for col in model_columns:
            if col not in input_encoded:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_columns]

        # Predict
        prob        = model.predict_proba(input_encoded)[0][1]
        prediction  = int(prob >= threshold)
        loyalty     = 1 - prob

        # Gauge
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = round(loyalty * 100, 1),
            delta = {"reference": 50, "valueformat": ".1f"},
            title = {"text": "Loyalty Score", "font": {"size": 18}},
            number= {"suffix": "%", "font": {"size": 32}},
            gauge = {
                "axis":  {"range": [0, 100], "tickwidth": 1},
                "bar":   {"color": pink if prediction else "#90EE90"},
                "steps": [
                    {"range": [0,  40], "color": "#FFB6C1"},
                    {"range": [40, 70], "color": "#FFDAB9"},
                    {"range": [70,100], "color": "#C1FFC1"},
                ],
                "threshold": {
                    "line":  {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Verdict
        if prediction:
            st.markdown(f"""
            <div class="churn-alert">
                ⚠️ <strong>High Churn Risk</strong><br>
                This customer has a <strong>{prob*100:.1f}%</strong> probability of churning.
                Immediate retention action recommended.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                ✅ <strong>Low Churn Risk</strong><br>
                This customer has only a <strong>{prob*100:.1f}%</strong> churn probability.
                Continue standard engagement.
            </div>""", unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob*100:.1f}%")
        m2.metric("Loyalty Score",     f"{loyalty*100:.1f}%")
        m3.metric("Tenure",            f"{tenure} months")

        # ── SHAP explanation ──────────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Why this Prediction? (SHAP)")

        shap_values = explainer.shap_values(input_encoded)

        shap_df = pd.DataFrame({
            "Feature":    input_encoded.columns,
            "SHAP Value": shap_values[0]
        }).sort_values("SHAP Value", key=abs, ascending=False).head(8)

        fig2, ax = plt.subplots(figsize=(7, 4))
        colors = [pink if v > 0 else blue for v in shap_df["SHAP Value"]]
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on churn probability)")
        ax.set_title("Top Feature Contributions")
        ax.invert_yaxis()
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

        st.caption(
            "🔴 Pink bars push toward churn. 🔵 Blue bars push toward retention. "
            "Bar length = magnitude of impact."
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#adb5bd; font-size:13px;'>
    ChurnShield · XGBoost + SHAP · Telco Customer Churn Dagitaset ·
    Built by <a href="https://github.com/nishi-0212" style="color:#F48FB1;">Nishi Vishwakarma</a>
</div>
""", unsafe_allow_html=True)
