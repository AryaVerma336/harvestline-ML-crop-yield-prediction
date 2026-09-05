"""
Harvestline — Training Script
Trains the Random Forest Regressor pipeline on FAO yield data, evaluates performance,
and exports the serialized pipeline artifact (.pkl).
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "yield_df.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "crop_yield_pipeline.pkl")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset file not found at {DATA_PATH}")
        sys.exit(1)

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records across {df['Area'].nunique()} countries and {df['Item'].nunique()} crops.")

    # Feature Engineering
    START_YEAR = 1990
    df["rain_temp_interaction"] = df["average_rain_fall_mm_per_year"] * df["avg_temp"]
    df["pesticide_per_rainfall"] = df["pesticides_tonnes"] / (df["average_rain_fall_mm_per_year"] + 1e-5)
    df["years_since_start"] = df["Year"] - START_YEAR

    num_features = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "rain_temp_interaction",
        "pesticide_per_rainfall",
        "years_since_start",
    ]
    cat_features = ["Area", "Item"]
    target = "hg/ha_yield"

    X = df[num_features + cat_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("rf", rf_model)])

    print("Training Random Forest Regressor Pipeline...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print("\n--- Model Evaluation Results ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.2f} hg/ha")
    print(f"MAE:      {mae:.2f} hg/ha")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel pipeline saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
