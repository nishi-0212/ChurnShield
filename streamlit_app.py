# -- CODE BLOCK START --
# All logic is untouched; only questions have been rewritten for clarity.

# Replace all your existing sidebar widgets with this updated section:

st.sidebar.header("📝 Customer Information")

# Reset button functionality
if st.sidebar.button("🔄 Reset All Inputs"):
    for key, value in default_inputs.items():
        st.session_state[key] = value
    st.rerun()

# Get current values
values = {key: st.session_state.get(key, default_inputs[key]) for key in default_inputs}

gender = st.sidebar.selectbox(
    "What is your gender?", ["Male", "Female"], 
    index=["Male", "Female"].index(values["Gender"])
)

senior_citizen_label = st.sidebar.selectbox(
    "Are you a senior citizen?", ["No", "Yes"],
    index=["No", "Yes"].index(values["SeniorCitizen"])
)
senior_citizen = 1 if senior_citizen_label == "Yes" else 0

partner = st.sidebar.selectbox(
    "Do you have a partner?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["Partner"])
)

dependents = st.sidebar.selectbox(
    "Do you have any dependents (like children)?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["Dependents"])
)

tenure = st.sidebar.slider(
    "How many months have you been with us?", 0, 72, 
    int(values["tenure"])
)

phone_service = st.sidebar.selectbox(
    "Do you have a phone connection with us?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["PhoneService"])
)

multiple_lines = st.sidebar.selectbox(
    "Do you have multiple phone lines?", ["Yes", "No", "No phone service"], 
    index=["Yes", "No", "No phone service"].index(values["MultipleLines"])
)

internet_service = st.sidebar.selectbox(
    "What type of internet service do you use?", ["DSL", "Fiber optic", "No"], 
    index=["DSL", "Fiber optic", "No"].index(values["InternetService"])
)

online_security = st.sidebar.selectbox(
    "Do you use our online security service?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["OnlineSecurity"])
)

online_backup = st.sidebar.selectbox(
    "Do you use our online backup service?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["OnlineBackup"])
)

device_protection = st.sidebar.selectbox(
    "Do you have device protection with us?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["DeviceProtection"])
)

tech_support = st.sidebar.selectbox(
    "Do you use our tech support service?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["TechSupport"])
)

streaming_tv = st.sidebar.selectbox(
    "Do you stream TV using our service?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["StreamingTV"])
)

streaming_movies = st.sidebar.selectbox(
    "Do you stream movies using our service?", ["Yes", "No", "No internet service"], 
    index=["Yes", "No", "No internet service"].index(values["StreamingMovies"])
)

contract = st.sidebar.selectbox(
    "What type of contract do you have?", ["Month-to-month", "One year", "Two year"], 
    index=["Month-to-month", "One year", "Two year"].index(values["Contract"])
)

paperless_billing = st.sidebar.selectbox(
    "Do you use paperless billing?", ["Yes", "No"], 
    index=["Yes", "No"].index(values["PaperlessBilling"])
)

payment_method = st.sidebar.selectbox(
    "How do you usually pay your bill?",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(values["PaymentMethod"])
)

monthly_charges = st.sidebar.number_input(
    "What is your monthly charge?", min_value=0.0, max_value=200.0, value=float(values["MonthlyCharges"])
)

total_charges = st.sidebar.number_input(
    "What is your total amount paid so far?", min_value=0.0, max_value=10000.0, value=float(values["TotalCharges"])
)
# -- CODE BLOCK END --

