
import pandas as pd

from src.config import TARGET_COLUMN


def clean_data(df):
    """Clean common issues in the Telco churn dataset."""
    df = df.copy()

    # Remove accidental spaces in column names
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean text values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str)
            else value
        )

    # Common Telco dataset issue: TotalCharges contains blank strings
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # Keep rows with a valid target
    df = df.dropna(subset=[TARGET_COLUMN])

    # Standardize common target formats
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
    })

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )
    df = df.dropna(subset=[TARGET_COLUMN])

    valid_labels = {0, 1}
    if not set(df[TARGET_COLUMN].unique()).issubset(valid_labels):
        raise ValueError(
            "Churn must contain binary labels: Yes/No or 1/0."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df.reset_index(drop=True)


def split_features_target(df):
    """Separate input features (X) and target (y)."""
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    # CustomerID is an identifier, not a useful predictive feature
    id_columns = [
        column for column in X.columns
        if column.lower() in {"customerid", "customer_id"}
    ]
    X = X.drop(columns=id_columns)

    return X, y
