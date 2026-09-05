"""
Harvestline — Lookup Generator Script
Generates lookup.json and trend.json from the dataset for frontend auto-fill
and historical trend visualizations.
"""
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "yield_df.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")
LOOKUP_PATH = os.path.join(MODEL_DIR, "lookup.json")
TREND_PATH = os.path.join(MODEL_DIR, "trend.json")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset file not found at {DATA_PATH}")
        return

    print(f"Reading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    lookup = {}
    trend = {}

    grouped = df.groupby(["Area", "Item"])

    for (area, item), group in grouped:
        key = f"{area}||{item}"

        lookup[key] = {
            "avg_rainfall": round(float(group["average_rain_fall_mm_per_year"].mean()), 2),
            "avg_pesticides": round(float(group["pesticides_tonnes"].mean()), 2),
            "avg_temp": round(float(group["avg_temp"].mean()), 2),
            "min_year": int(group["Year"].min()),
            "max_year": int(group["Year"].max()),
            "avg_yield": round(float(group["hg/ha_yield"].mean()), 1),
            "min_yield": float(group["hg/ha_yield"].min()),
            "max_yield": float(group["hg/ha_yield"].max()),
        }

        yearly_trend = {}
        for _, row in group.iterrows():
            yearly_trend[str(int(row["Year"]))] = round(float(row["hg/ha_yield"]), 1)

        trend[key] = yearly_trend

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(LOOKUP_PATH, "w") as f:
        json.dump(lookup, f, indent=2)

    with open(TREND_PATH, "w") as f:
        json.dump(trend, f, indent=2)

    print(f"Successfully generated:\n  - {LOOKUP_PATH}\n  - {TREND_PATH}")


if __name__ == "__main__":
    main()
