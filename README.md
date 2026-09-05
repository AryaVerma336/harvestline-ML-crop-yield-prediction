# 🌾 Harvestline — Machine Learning Crop Yield Prediction & Climate Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)](https://flask.palletsprojects.com/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"Every harvest starts as a line in the ledger."**

Harvestline is an end-to-end Machine Learning web application and analytical suite for predicting global agricultural crop yields per hectare (`hg/ha` and `tonnes/ha`). Powered by a trained **Random Forest Regressor** pipeline (`ColumnTransformer` + `RandomForestRegressor`) trained on 28,000+ FAO & World Bank historical records across 100+ countries and major crop varieties.

---

## ✨ Key Features

- 🧠 **Explainable AI (XAI) Feature Attribution**: Deconstructs every individual prediction to show how specific inputs (Temperature, Rainfall, Pesticide use, Climate interactions) pull the forecast above or below regional historical baselines.
- 🌦️ **Climate Scenario Simulator ("What-If" Analysis)**: Instant presets for climate change stress testing:
  - ☀️ **IPCC +1.5°C Warming** (+1.5°C temp, −5% rain)
  - 🔥 **Extreme Heatwave & Drought** (+3.5°C temp, −30% rain)
  - 🌧️ **Heavy Monsoon Deluge** (+35% rain, −1.0°C temp)
  - 🌱 **Optimal Yield** (Regional ideal growth recipe)
- 🎓 **Academic Model Evaluation Dashboard**: Interactive model benchmarking comparing **Random Forest** ($R^2 = 0.965$) against baseline **Decision Tree** ($R^2 = 0.884$) and **Linear Regression** ($R^2 = 0.712$), alongside an actual vs. predicted test set scatter plot.
- 🗺️ **Spatial Geographic Yield Map**: Embedded Leaflet.js interactive map visualizing worldwide crop yield distributions by country.
- 📊 **Historical Trend Marker**: Interactive Chart.js graph marking live predictions against 23 years (1990–2013) of recorded FAO crop yields.

---

## 📈 Model Performance & Benchmarking

The model was evaluated using 80/20 train-test splits on historical FAO harvest data:

| Model Algorithm | $R^2$ Score | RMSE ($\text{hg/ha}$) | RMSE ($\text{t/ha}$) | MAE ($\text{hg/ha}$) | MAE ($\text{t/ha}$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9650** | **14,820** | **1.48** | **8,340** | **0.83** | 🟢 **Production Pipeline** |
| Decision Tree Regressor | 0.8840 | 26,400 | 2.64 | 15,100 | 1.51 | 🟡 Baseline Benchmark |
| Linear Regression | 0.7120 | 41,200 | 4.12 | 28,900 | 2.89 | 🔴 Linear Baseline |

---

## 🧮 Domain Feature Engineering

The training pipeline constructs three engineered domain interaction features:

$$\text{rain\_temp\_interaction} = \text{average\_rain\_fall\_mm\_per\_year} \times \text{avg\_temp}$$

$$\text{pesticide\_per\_rainfall} = \frac{\text{pesticides\_tonnes}}{\text{average\_rain\_fall\_mm\_per\_year} + 10^{-5}}$$

$$\text{years\_since\_start} = \text{Year} - 1990$$

---

## 📁 Repository Structure

```
harvestline/
├── backend/
│   ├── app.py                      # Flask REST API & Web Application server
│   ├── requirements.txt            # Backend dependencies
│   ├── model/
│   │   ├── crop_yield_pipeline.pkl # Serialized Scikit-Learn Pipeline (70MB)
│   │   ├── lookup.json             # Per country & crop historical baseline lookups
│   │   └── trend.json              # 1990–2013 yearly crop yield series
│   ├── templates/
│   │   └── index.html              # Modern, responsive Web UI dashboard
│   └── static/
│       ├── style.css               # Design system & dark theme UI styles
│       ├── script.js              # Interactive logic (XAI, Map, Charts, Scenarios)
│       └── chart.umd.js            # Bundled Chart.js library
├── data/
│   └── yield_df.csv                # Processed training dataset (FAO & World Bank data)
├── notebooks/
│   └── Harvestline_Crop_Yield_Prediction_ML_Pipeline.ipynb  # Google Colab Notebook
├── scripts/
│   ├── train_model.py              # CLI training & evaluation script
│   └── generate_lookups.py         # JSON lookup generator from raw CSV
├── requirements.txt                # Global Python dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Documentation
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.10+
- `pip` & `venv`

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/harvestline.git
cd harvestline
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Flask Web Application
```bash
cd backend
PORT=5050 python app.py
```
Open **[http://localhost:5050](http://localhost:5050)** in your browser.

---

## 📓 Google Colab Notebook

You can inspect, train, and modify the Machine Learning model directly in Google Colab:
- Open [`notebooks/Harvestline_Crop_Yield_Prediction_ML_Pipeline.ipynb`](notebooks/Harvestline_Crop_Yield_Prediction_ML_Pipeline.ipynb) in Colab.
- Run all cells to perform EDA, feature engineering, model training, evaluation, and pipeline export.

To retrain the model via CLI:
```bash
python scripts/train_model.py
python scripts/generate_lookups.py
```

---

## 🔌 API Reference

### `POST /api/predict`
Calculates yield prediction and XAI feature contributions for given inputs.

**Request Payload:**
```json
{
  "area": "India",
  "item": "Wheat",
  "year": 2013,
  "rainfall": 1080.0,
  "temp": 24.5,
  "pesticides": 50000.0
}
```

**Response Output:**
```json
{
  "prediction_hg_ha": 24787.2,
  "prediction_t_ha": 2.479,
  "band_t_ha": [2.20, 2.68],
  "delta_pct": -6.6,
  "xai_breakdown": [
    { "feature": "Regional Baseline Yield", "impact_t_ha": 2.476, "type": "base" },
    { "feature": "Temperature Adjustment", "impact_t_ha": 0.011, "type": "delta" },
    { "feature": "Rainfall Adjustment", "impact_t_ha": -0.034, "type": "delta" },
    { "feature": "Pesticide Use Adjustment", "impact_t_ha": 0.003, "type": "delta" },
    { "feature": "Climate Interaction Effect", "impact_t_ha": 0.023, "type": "delta" }
  ]
}
```

### `GET /api/evaluation`
Returns algorithm comparison statistics ($R^2$, RMSE, MAE) and scatter plot points.

### `GET /api/map_data?item=Wheat&year=2013`
Returns estimated crop yields across all countries for global spatial map rendering.

---

## 📜 Dataset Source & License

Data sourced from public records compiled by the **Food and Agriculture Organization (FAO)** and **The World Bank** (1990–2013).

Distributed under the MIT License.
