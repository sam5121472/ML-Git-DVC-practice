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



# =====================================================
# 4. FEATURE SELECTION (TOP 6 FEATURES)
# =====================================================

print("\n========== FEATURE SELECTION ==========")


# Separate features and target

X = df.drop(
    "has_heart_disease",
    axis=1
)

y = df["has_heart_disease"]



# Select top 6 features

selector = SelectKBest(
    score_func=f_classif,
    k=6
)


X_selected = selector.fit_transform(
    X,
    y
)



# Selected feature names

selected_features = X.columns[
    selector.get_support()
]


print("\nTop 6 Selected Features:")

for feature in selected_features:
    print(feature)



# Create selected dataset

selected_df = pd.DataFrame(
    X_selected,
    columns=selected_features
)


selected_df["has_heart_disease"] = y



print("\nSelected Dataset:")
print(selected_df.head())


print("\nSelected Dataset Shape:")
print(selected_df.shape)



# Save selected features dataset

selected_path = r"data/selected_features.csv"


selected_df.to_csv(
    selected_path,
    index=False
)


print("\nFeature selection completed successfully!")



#%%

# =====================================================
# 5. TRAIN TEST SPLIT + FEATURE SCALING
# =====================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print("\n========== TRAIN TEST SPLIT & SCALING ==========")


# Load selected features dataset

file_path = r"data/selected_features.csv"

df = pd.read_csv(file_path)


print("Selected Features Dataset:")
print(df.head())

print("\nShape:")
print(df.shape)



# ---------------------------------
# Separate Features and Target
# ---------------------------------

X = df.drop(
    "has_heart_disease",
    axis=1
)

y = df["has_heart_disease"]



# ---------------------------------
# Train Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)



# ---------------------------------
# Feature Scaling
# ---------------------------------

scaler = StandardScaler()


# Fit only on training data
X_train_scaled = scaler.fit_transform(
    X_train
)


# Transform test data
X_test_scaled = scaler.transform(
    X_test
)



# Convert back to DataFrame

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)


X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)



print("\nScaled Training Data:")
print(X_train_scaled.head())


print("\nScaled Testing Data:")
print(X_test_scaled.head())



# ---------------------------------
# Save Scaled Data
# ---------------------------------

X_train_scaled.to_csv(
    r"data/X_train_scaled.csv",
    index=False
)


X_test_scaled.to_csv(
    r"data/X_test_scaled.csv",
    index=False
)


y_train.to_csv(
    r"data/y_train.csv",
    index=False
)


y_test.to_csv(
    r"data/y_test.csv",
    index=False
)


print("\nScaling and train-test split completed successfully!")



#%%

# =====================================================
# 6. MODEL TRAINING (Random Forest + Logistic Regression)
# =====================================================

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


print("\n========== MODEL TRAINING ==========")


# ---------------------------------
# Load Train Test Data
# ---------------------------------

X_train = pd.read_csv(
    r"data/X_train_scaled.csv"
)

X_test = pd.read_csv(
    r"data/X_test_scaled.csv"
)


y_train = pd.read_csv(
    r"data/y_train.csv"
).values.ravel()


y_test = pd.read_csv(
    r"data/y_test.csv"
).values.ravel()



# =====================================================
# 1. Logistic Regression
# =====================================================

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# Train model
lr_model.fit(
    X_train,
    y_train
)


# Prediction
lr_pred = lr_model.predict(
    X_test
)



# Evaluation

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

lr_precision = precision_score(
    y_test,
    lr_pred
)

lr_recall = recall_score(
    y_test,
    lr_pred
)

lr_f1 = f1_score(
    y_test,
    lr_pred
)



print("\nLogistic Regression Results")

print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1 Score:", lr_f1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        lr_pred
    )
)



# =====================================================
# 2. Random Forest
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model

rf_model.fit(
    X_train,
    y_train
)


# Prediction

rf_pred = rf_model.predict(
    X_test
)



# Evaluation

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)



print("\nRandom Forest Results")

print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred
    )
)



# =====================================================
# Model Comparison
# =====================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        lr_accuracy,
        rf_accuracy
    ],

    "Precision": [
        lr_precision,
        rf_precision
    ],

    "Recall": [
        lr_recall,
        rf_recall
    ],

    "F1 Score": [
        lr_f1,
        rf_f1
    ]

})


print("\n========== MODEL COMPARISON ==========")

print(results)



# Save Results

results.to_csv(
    r"data/model_results.csv",
    index=False
)


print("\nModel training completed successfully!")