import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("First 10 Records:")
print(df.head(10))
print("\n" + "-"*50 + "\n")

print("Missing Values Check:")
print(df.isnull().sum())
print("\n" + "-"*50 + "\n")

print("Class Distribution:")
print(df['target'].value_counts())
print("Classes:", data.target_names)
print("\n" + "-"*50 + "\n")

print("Dataset Summary:")
print(df.info())
print("\n" + "-"*50 + "\n")

print("Statistical Analysis:")
print(df.describe())
