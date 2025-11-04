import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib
import numpy as np

# ---- Load Data ----
# Adjust path as needed (currently assumes cbb25.csv exists in /data)
DATA_PATH = "data/cbb25.csv"
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.columns)

# ---- Baseline Training ----
print("\nStarting baseline training...")

# Create a simple binary target: team had > half of games won
if 'W' in df.columns and 'G' in df.columns:
    df['win_flag'] = (df['W'] > (df['G'] / 2)).astype(int)
    target_col = 'win_flag'
else:
    raise RuntimeError("No 'W' or 'G' columns found — update target logic.")

# Select numeric columns only
numeric = df.select_dtypes(include=[np.number]).copy()
X = numeric.drop(columns=[target_col])
y = numeric[target_col]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassifier metrics:")
print(" Accuracy:", accuracy_score(y_test, y_pred))
print(" F1:", f1_score(y_test, y_pred, zero_division=0))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/baseline_classifier.joblib")
print("\nSaved classifier -> models/baseline_classifier.joblib")

# Optional regression model (predicting total Wins)
if 'W' in numeric.columns:
    X_reg = numeric.drop(columns=['W'])
    y_reg = numeric['W']
    Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, random_state=42, test_size=0.2)
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(Xtr, ytr)
    ypred_reg = reg.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, ypred_reg))
    r2 = r2_score(yte, ypred_reg)

    print("\nRegressor metrics:")
    print(" RMSE:", rmse)
    print(" R2:", r2)
    joblib.dump(reg, "models/baseline_regressor.joblib")
    print("Saved regressor -> models/baseline_regressor.joblib")
