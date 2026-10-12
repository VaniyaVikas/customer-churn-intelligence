
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATA_FILE = RAW_DATA_DIR / "customer_churn.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "customer_churn_processed.csv"

# Model and report paths
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Target column
TARGET_COLUMN = "Churn"

# Reproducibility
RANDOM_STATE = 42

# Create required output directories
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    METRICS_PATH.parent,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
