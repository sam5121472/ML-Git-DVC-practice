#%%
import pandas as pd

# Load dataset
file_path = r"data/heart_disease_risk_2026.csv"

df = pd.read_csv(file_path)

# Display first 5 rows
print(df.head())

# Display dataset shape
print("Shape:", df.shape)
