
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR
from src.data_loader import load_data
from src.preprocessing import clean_data

df = clean_data(load_data())

# Churn rate by categorical feature
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for column in categorical_columns:
    if df[column].nunique() > 20:
        continue

    rates = (
        df.groupby(column, observed=False)["Churn"]
        .mean()
        .sort_values(ascending=False)
    )

    print(f"\nChurn rate by {column}:")
    print((rates * 100).round(2))

    rates.plot(kind="bar", figsize=(8, 4))
    plt.title(f"Churn Rate by {column}")
    plt.ylabel("Churn rate (0 to 1)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feature_{column}.png")
    plt.show()
