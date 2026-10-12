
from src.data_loader import load_data, inspect_data
from src.preprocessing import clean_data

df = load_data()

print("First 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nCleaned data:")
cleaned = clean_data(df)
inspect_data(cleaned)
