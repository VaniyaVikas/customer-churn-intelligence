
import pandas as pd

from src.preprocessing import clean_data, split_features_target


def test_clean_data_converts_churn_labels():
    df = pd.DataFrame({
        "customerID": ["A1", "A2", "A3"],
        "tenure": [1, 5, 10],
        "TotalCharges": ["20.5", " ", "100.0"],
        "Churn": ["Yes", "No", "Yes"],
    })

    cleaned = clean_data(df)

    assert cleaned["Churn"].tolist() == [1, 0, 1]
    assert cleaned["TotalCharges"].isna().sum() == 1


def test_split_features_target_drops_customer_id():
    df = pd.DataFrame({
        "customerID": ["A1", "A2"],
        "tenure": [2, 8],
        "Churn": [1, 0],
    })

    X, y = split_features_target(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y.tolist() == [1, 0]
