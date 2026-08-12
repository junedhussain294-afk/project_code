import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

sns.countplot(x='target', data=df)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

df.hist(figsize=(20,20))
plt.suptitle("Feature Histograms")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df.iloc[:, :5])
plt.title("Box Plot of First 5 Features")
plt.show()

# Optimized Pair Plot (only 3 features)
sns.pairplot(df[['mean radius','mean texture','mean area','target']], hue="target")
plt.show()
