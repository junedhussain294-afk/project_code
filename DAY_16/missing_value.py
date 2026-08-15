import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 1. Check missing values
print("Missing values per column:\n", df.isnull().sum())

# 2. Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('target', axis=1))

# Convert back to DataFrame for clarity
X_scaled_df = pd.DataFrame(X_scaled, columns=data.feature_names)

# 3. Normalize selected features (example: first 5 features)
normalizer = MinMaxScaler()
X_norm = normalizer.fit_transform(X_scaled_df.iloc[:, :5])
X_norm_df = pd.DataFrame(X_norm, columns=data.feature_names[:5])

print("\nNormalized sample (first 5 features):\n", X_norm_df.head())

# 4. Remove duplicate records (if any)
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]

print(f"\nDuplicates removed: {before - after}")
print("Final dataset shape:", df.shape)
