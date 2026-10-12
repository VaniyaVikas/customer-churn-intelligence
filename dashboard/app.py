
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH
from src.predict import load_model
from src.retention import recommend_retention


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Intelligence")
st.write("Predict churn risk and explore customer retention actions.")

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python -m src.train")
    st.stop()

@st.cache_resource
def get_model():
    return load_model()


model = get_model()

with st.form("customer_form"):
    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        probability = float(
            model.predict_proba(pd.DataFrame([customer]))[0, 1]
        )
        result = recommend_retention(customer, probability)

        st.subheader("Prediction Result")
        metric1, metric2 = st.columns(2)

        metric1.metric("Churn Probability", f"{probability:.1%}")
        metric2.metric("Risk Level", result["risk_level"])

        if probability >= 0.5:
            st.warning("The model predicts that this customer may churn.")
        else:
            st.success("The model predicts that this customer may stay.")

        st.write("### Suggested Retention Actions")
        for recommendation in result["recommendations"]:
            st.write(f"- {recommendation}")

        st.caption(
            "Predictions are estimates, not guarantees. "
            "The 50% threshold is a starting point and should be validated."
        )

    except (ValueError, KeyError) as exc:
        st.error(f"Prediction failed: {exc}")
