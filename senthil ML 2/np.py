import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("qs-world-rankings-2025.csv")

print("Dataset Loaded Successfully\n")
print(df.head())
print(df.tail())

df.columns = df.columns.str.strip()

print("\nColumn Names:")
print(df.columns)
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df = df.drop_duplicates()

print("\nStatistical Summary:")
print(df.describe())

df[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
df[numeric_cols].boxplot()
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

target_candidates = [col for col in numeric_cols if "score" in col.lower()]

if len(target_candidates) == 0:
    raise ValueError("No score column found in dataset!")

target = target_candidates[0]
print("\nSelected Target Column:", target)

corr_values = df[numeric_cols].corr()[target].sort_values(ascending=False)

print("\nCorrelation with Target:")
print(corr_values)

features = corr_values.index[1:6].tolist()

print("\nSelected Features:", features)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R² Score:", r2)

print("\nModel Coefficients:")
for col, coef in zip(features, model.coef_):
    print(col, ":", coef)

print("\nTraining Completed Successfully!")
