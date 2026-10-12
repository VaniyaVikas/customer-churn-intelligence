
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR
from src.data_loader import load_data
from src.preprocessing import clean_data

df = clean_data(load_data())

# Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_distribution.png")
plt.show()

# Churn percentage
print("Churn rate (%):")
print((df["Churn"].mean() * 100).round(2))

# Numeric feature distributions
numeric_columns = df.select_dtypes(include="number").columns.tolist()

for column in numeric_columns:
    if column == "Churn":
        continue

    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x=column, hue="Churn", kde=True)
    plt.title(f"{column} Distribution by Churn")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{column}_distribution.png")
    plt.show()

# Churn by contract type
if "Contract" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Churn by Contract Type")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_contract.png")
    plt.show()
