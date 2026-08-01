#%%

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


# =====================================================
# 1. LOAD DATASET
# =====================================================

file_path = r"data/heart_disease_risk_2026.csv"

df = pd.read_csv(file_path)

print("Original Dataset")
print(df.head())

print("\nShape:")
print(df.shape)



# =====================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================

print("\n========== EDA ==========")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# Missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())


# Numerical summary
print("\nNumerical Summary:")
print(df.describe())


# Categorical summary
categorical_columns = df.select_dtypes(include=["object"]).columns

if len(categorical_columns) > 0:
    print("\nCategorical Summary:")
    print(df[categorical_columns].describe())
else:
    print("\nNo categorical object columns found.")


# Unique values
print("\nUnique Values:")
for column in df.columns:
    print(column, ":", df[column].nunique())


# Target distribution
print("\nHeart Disease Distribution:")
print(df["has_heart_disease"].value_counts())



# =====================================================
# 3. DATA PREPROCESSING
# =====================================================

print("\n========== DATA PREPROCESSING ==========")


# Remove duplicates
df = df.drop_duplicates()


# Handle missing values

# Numerical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns


for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())


# Categorical columns
categorical_columns = df.select_dtypes(
    include=["object"]
).columns


for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])



# Remove ID column
if "patient_id" in df.columns:
    df = df.drop(columns=["patient_id"])



# Encode categorical variables
df = pd.get_dummies(
    df,
    drop_first=True
)



print("\nProcessed Dataset:")
print(df.head())

print("\nProcessed Shape:")
print(df.shape)


print("\nMissing Values After Processing:")
print(df.isnull().sum().sum())



# Save processed dataset

processed_path = r"data/processed_heart_disease.csv"

df.to_csv(
    processed_path,
    index=False
)


print("\nProcessed dataset saved!")



