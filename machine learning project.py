#%%
import pandas as pd

# Load dataset
file_path = r"data/heart_disease_risk_2026.csv"

df = pd.read_csv(file_path)

# Display first 5 rows
print(df.head())

# Display dataset shape
print("Shape:", df.shape)

#EDA


# -----------------------------
# Basic Information
# -----------------------------
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Duplicate Records
# -----------------------------
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# Statistical Summary
# -----------------------------
print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include="object"))

# -----------------------------
# Unique Values
# -----------------------------
print("\nUnique Values in Each Column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

# -----------------------------
# Target Variable Distribution
# -----------------------------
print("\nHeart Disease Distribution:")
print(df["has_heart_disease"].value_counts())
