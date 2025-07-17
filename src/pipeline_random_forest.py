import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from pipelines.base_pipeline import build_preprocessor
from utils.data_utils import load_data, split_by_year

# === CONFIG ===
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURR_PATH, '..', 'data', 'processed.csv.gz')
TARGET_COL = "AveragePrice"
MODEL_OUTPUT_PATH = os.path.join(CURR_PATH, '..', 'models', 'random_forest_model.pkl')

# === Load Data ===
print("Loading data...")
df = load_data(DATA_PATH, TARGET_COL)

# === Time-Based Split ===
X_train, X_test, y_train, y_test = split_by_year(df, TARGET_COL, split_year=2020)

# === Preprocessing ===
print("Building pipeline...")

preprocessor = build_preprocessor(X_train)

# === Full Model Pipeline ===
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))])

# === Train Model ===
print("Training model...")
model_pipeline.fit(X_train, y_train)

# === Evaluate ===
print("Evaluating model...")
y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2: {r2:.3f}")
print(f"RMSE: {rmse:,.2f}")

# === Save Model ===
print(f"Saving model to {MODEL_OUTPUT_PATH}...")
joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)

print("Done")
