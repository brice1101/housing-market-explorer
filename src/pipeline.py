import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# === CONFIG ===
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURR_PATH, '..', 'data', 'processed.csv.gz')
TARGET_COL = "AveragePrice"
MODEL_OUTPUT_PATH = os.path.join(CURR_PATH, '..', 'models', 'linear_model.pkl')

# === Load Data ===
print("Loading data...")
df = pd.read_csv(DATA_PATH, compression='gzip')

# === Drop rows with missing target ===
df = df.dropna(subset=[TARGET_COL])

# === Define Features ===
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Identify column types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category'])

# === Train/Test Split ===
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

# === Preprocessing ===
print("Building pipeline...")

# Pipeline for numeric features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='mean')),
    ("scaler", StandardScaler())])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features)])

# === Full Model Pipeline ===
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())])

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
