import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
MODEL_PATH = BASE_DIR / "models" / "telco_churn_model.pkl"
model = joblib.load(MODEL_PATH)

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# App title
st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Enter customer information below to predict the likelihood of customer churn."
)

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

st.header("Additional Services")

col1, col2 = st.columns(2)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

st.header("Billing Information")

col1, col2 = st.columns(2)

with col1:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.25,
        max_value=118.75,
        value=70.35
    )

with col2:
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=8684.80,
        value=1394.55
    )


st.divider()

if st.button("🔮 Predict Churn", type="primary", key="predict_churn"):

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0, 1]

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if probability >= 0.70:
         st.error("🔴 High Churn Risk")
    elif probability >= 0.36:
         st.warning("🟠 Moderate Churn Risk")
    else:
         st.success("🟢 Lower Churn Risk")

    st.progress(float(probability))

    if prediction == "Yes":
        st.write(
            "The model predicts that this customer is likely to churn "
            "based on the information provided."
        )
    else:
        st.write(
            "The model predicts that this customer is less likely to churn "
            "based on the information provided."
        )

