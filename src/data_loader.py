
import pandas as pd

from src.config import DATA_FILE, TARGET_COLUMN


def load_data(file_path=DATA_FILE):
    """Load customer churn data from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place customer_churn.csv inside data/raw/."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    df.columns = df.columns.str.strip()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def inspect_data(df):
    """Print a basic dataset summary."""
    print("\nDataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))


if __name__ == "__main__":
    data = load_data()
    inspect_data(data)
