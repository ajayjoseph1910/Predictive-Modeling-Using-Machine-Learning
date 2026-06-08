# ==========================================
# PREDICTIVE MODELING USING MACHINE LEARNING
# COVID-19 COUNTRY-WISE DATASET
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("country_wise_latest.csv")

print("="*60)
print("FIRST 5 ROWS")
print(df.head())

print("\n" + "="*60)
print("DATASET INFORMATION")
print(df.info())

print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print(df.describe())

# ==========================================
# DATA CLEANING
# ==========================================

print("\n" + "="*60)
print("MISSING VALUES")
print(df.isnull().sum())

print("\n" + "="*60)
print("DUPLICATES")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ==========================================
# FEATURE SELECTION
# Predict Deaths
# ==========================================

X = df[['Confirmed', 'Recovered', 'Active']]
y = df['Deaths']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))

# ==========================================
# LINEAR REGRESSION MODEL
# ==========================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

# ==========================================
# LINEAR REGRESSION EVALUATION
# ==========================================

print("\n" + "="*60)
print("LINEAR REGRESSION RESULTS")

print("MAE:",
      mean_absolute_error(y_test, lr_predictions))

print("MSE:",
      mean_squared_error(y_test, lr_predictions))

print("R2 Score:",
      r2_score(y_test, lr_predictions))

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# ==========================================
# RANDOM FOREST EVALUATION
# ==========================================

print("\n" + "="*60)
print("RANDOM FOREST RESULTS")

print("MAE:",
      mean_absolute_error(y_test, rf_predictions))

print("MSE:",
      mean_squared_error(y_test, rf_predictions))

print("R2 Score:",
      r2_score(y_test, rf_predictions))

# ==========================================
# VISUALIZATION 1
# ACTUAL VS PREDICTED
# LINEAR REGRESSION
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(y_test,
            lr_predictions)

plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Linear Regression: Actual vs Predicted")

plt.show()

# ==========================================
# VISUALIZATION 2
# ACTUAL VS PREDICTED
# RANDOM FOREST
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(y_test,
            rf_predictions)

plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Random Forest: Actual vs Predicted")

plt.show()

# ==========================================
# VISUALIZATION 3
# FEATURE CORRELATION
# ==========================================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ==========================================
# VISUALIZATION 4
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(8,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance - Random Forest")

plt.show()

# ==========================================
# VISUALIZATION 5
# MODEL COMPARISON
# ==========================================

lr_r2 = r2_score(y_test, lr_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

models = ['Linear Regression',
          'Random Forest']

scores = [lr_r2,
          rf_r2]

plt.figure(figsize=(8,5))

sns.barplot(
    x=models,
    y=scores
)

plt.ylabel("R2 Score")
plt.title("Model Comparison")

plt.show()

# ==========================================
# FINAL CONCLUSION
# ==========================================

print("\n" + "="*60)

if rf_r2 > lr_r2:
    print("Random Forest performed better.")
else:
    print("Linear Regression performed better.")

print("\nProject Completed Successfully!")