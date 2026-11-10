
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Create models folder
os.makedirs("models", exist_ok=True)

print("Loading Kaggle dataset...")

df = pd.read_csv("data/city_day.csv")

# Input features
features = [
    "PM2.5", "PM10", "NO", "NO2",
    "NOx", "NH3", "CO", "SO2", "O3"
]

target = "AQI"

# Verify columns
missing = [c for c in features + [target] if c not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# Convert values to numeric
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid target rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[target])

# Keep valid AQI range
df = df[df[target].between(0, 500)]

X = df[features]
y = df[target]

print("Dataset shape:", df.shape)
print("Preparing training data...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# ANN model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("ann", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=42
    ))
])

print("Training Artificial Neural Network...")

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save trained model
joblib.dump({
    "model": model,
    "features": features
}, "models/air_quality_model.pkl")

print("\nModel saved successfully!")
print("Location: models/air_quality_model.pkl")
print("===================================")
