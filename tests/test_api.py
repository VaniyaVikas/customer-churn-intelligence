
import pytest
from pydantic import ValidationError

from api.schemas import CustomerInput


def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
    }


def test_customer_input_accepts_valid_data():
    customer = CustomerInput(**valid_customer())

    assert customer.tenure == 12
    assert customer.gender == "Female"


def test_customer_input_rejects_negative_tenure():
    data = valid_customer()
    data["tenure"] = -1

    with pytest.raises(ValidationError):
        CustomerInput(**data)


def test_customer_input_rejects_extra_fields():
    data = valid_customer()
    data["unexpected_field"] = "invalid"

    with pytest.raises(ValidationError):
        CustomerInput(**data)
