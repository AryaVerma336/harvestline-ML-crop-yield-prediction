# 🌾 Harvestline — Machine Learning Crop Yield Prediction & Agro-Climatic Intelligence
## *Academic Defense Edition · India-Centric Mandi Units & Global Agricultural Analytics*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)](https://flask.palletsprojects.com/)
[![R2 Score](https://img.shields.io/badge/R%C2%B2%20Score-0.987-brightgreen.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"Every harvest starts as a line in the ledger."**

Harvestline is an end-to-end Machine Learning web application, climate risk simulator, and agro-climatic intelligence platform for forecasting agricultural crop yields per hectare and per acre. Powered by a production **Random Forest Regressor pipeline** (`ColumnTransformer` + `StandardScaler` + `OneHotEncoder` + `RandomForestRegressor`) trained on **13,734 clean, deduplicated historical records** from the **UN Food and Agriculture Organization (FAO)** and **The World Bank** across 101 countries and 10 major crop varieties (1990–2013).

---

## 🌟 Key Highlights & Academic Innovations

- 🇮🇳 **India-Centric Agricultural Intelligence & Mandi Units**:
  - Displays outputs in **Quintals per Hectare (q/ha)** and **Quintals per Acre (q/acre)**, directly aligning with Indian APMC Mandi trading standards and Minimum Support Price (MSP) declarations.
  - Interactive presets for Indian cropping cycles: **Kharif (Monsoon)**, **Rabi (Winter)**, and **Zaid (Summer)**.
  - Agro-climatic regional presets: **Indo-Gangetic Plain (Punjab/Haryana)**, **Eastern Delta (Bengal/Odisha)**, **Central Deccan (MP/Maharashtra)**, and **Southern Peninsula (Tamil Nadu/Kerala)**.
- 📊 **Multi-Model Academic Benchmarking Dashboard**:
  - Direct comparative benchmarking of 5 algorithms: **Random Forest** ($R^2 = 0.987$), **XGBoost** ($R^2 = 0.979$), **Single Decision Tree** ($R^2 = 0.884$), **Linear Regression (OLS)** ($R^2 = 0.766$), and **Support Vector Regressor (SVR)** ($R^2 = -0.175$).
  - Evaluated with 80/20 train-test splits and **5-Fold Cross-Validation**.
  - **Paired Bootstrap Significance Testing** ($B = 2,000$ resamples) demonstrating Random Forest's advantage over linear baselines is statistically significant ($p < 0.001$, $95\%\text{ CI } [2.65, 3.01]\text{ t/ha}$).
- 🌐 **India vs. Global Crop Yield Gap Comparison**:
  - Interactive Chart.js comparative graph contrasting Indian crop yields against global benchmarks across all 8 Indian crops in the dataset (revealing India's world-leading tapioca productivity at 278 q/ha).
- 🌦️ **Climate Change Stress Simulator ("What-If" Analysis)**:
  - Side-by-side scenario comparison graph stress-testing crop yields against:
    - ☀️ **IPCC +1.5°C Global Warming** (+1.5°C temp, −5% rainfall)
    - 🔥 **El Niño Heatwave & Drought** (+3.5°C temp, −30% rainfall deficit)
    - 🌧️ **Monsoon Deluge / Excess Precipitation** (+35% rainfall, −1.0°C temp)
    - 🌱 **Optimal Agro-Climatic Window** (Ideal thermal & moisture conditions)
- 🧠 **Explainable AI (XAI) Feature Attribution**:
  - Marginal Contribution Breakdown deconstructing every individual forecast into: $\text{Baseline} + \Delta\text{Temp} + \Delta\text{Rain} + \Delta\text{Pest} + \Delta\text{Interactions}$.
- 🎓 **Built-in Teacher Review & Viva Defense Guide**:
  - Interactive cheatsheet and FAQ in the web app answering the top 10 examination questions with full mathematical and agronomic explanations. See [TEACHER_REVIEW_GUIDE.md](TEACHER_REVIEW_GUIDE.md).

---

## 📈 Model Performance & Comparative Benchmark

| Model Algorithm | $R^2$ Score | RMSE ($\text{hg/ha}$) | RMSE ($\text{t/ha}$) | RMSE ($\text{q/ha}$) | MAE ($\text{t/ha}$) | 5-Fold CV RMSE | Model Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🟢 **Random Forest Regressor** | **0.9874** | **8,558** | **0.86** | **8.56** | **0.44** | **0.88 ± 0.09** | 🏆 **Production Champion Pipeline** |
| 🟡 **XGBoost Regressor** | 0.9792 | 10,972 | 1.10 | 10.97 | 0.69 | 1.08 ± 0.04 | ⚡ Gradient Boosting Alternative |
| 🟡 **Single Decision Tree** | 0.8840 | 26,400 | 2.64 | 26.40 | 1.51 | 2.72 ± 0.12 | 🌳 Unbagged Tree Baseline |
| 🟠 **Linear Regression (OLS)** | 0.7657 | 36,865 | 3.69 | 36.87 | 2.61 | 3.70 ± 0.06 | 📐 Linear Baseline Benchmark |
| 🔴 **Support Vector Regressor (SVR)**| -0.1747 | 82,546 | 8.25 | 82.55 | 5.08 | 8.28 ± 0.10 | ❌ Sparse One-Hot Distance Failure |
| ⚪ **Dummy Mean Predictor** | 0.0000 | 76,161 | 7.62 | 76.16 | 5.63 | 7.62 ± 0.05 | ➖ Naïve Training Mean Floor |

---

## 🇮🇳 Indian Agricultural Context & Mandi Units

### Unit Conversion Table

$$\text{1 Metric Tonne} = 10\text{ Quintals} = 1,000\text{ Kilograms}$$
$$\text{1 Hectare} \approx 2.47105\text{ Acres} = 10,000\text{ m}^2$$
$$\text{1 Tonne per Hectare (t/ha)} = 10\text{ q/ha} \approx 4.047\text{ q/acre}$$
$$\text{10,000 Hectograms per Hectare (hg/ha)} = 1\text{ Tonne/ha}$$

### Indian Crop Yield Benchmark (from FAO Records)

| Crop Variety | Local Name | Season | India Avg ($\text{q/ha}$) | Global Avg ($\text{q/ha}$) | Ratio to World | Key Producing States |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Cassava** | कसावा / साबूदाना | Tropical | **278.2** | 104.1 | **267.2%** 🟢 | Tamil Nadu (Salem sago cluster), Kerala |
| **Potatoes** | आलू (Aalu) | Rabi | **183.5** | 184.4 | **99.5%** 🟢 | Uttar Pradesh, West Bengal, Bihar, Punjab |
| **Wheat** | गेहूं (Kanak) | Rabi | **28.3** | 32.6 | **86.8%** 🟡 | Punjab, Haryana, UP, Madhya Pradesh |
| **Rice, paddy** | धान / चावल | Kharif | **31.8** | 39.2 | **81.1%** 🟡 | West Bengal, UP, Punjab, Andhra Pradesh |
| **Sweet potatoes**| शकरकंद | Kharif/Rabi | **89.8** | 105.8 | **84.9%** 🟡 | Odisha, West Bengal, UP, Kerala |
| **Soybeans** | सोयाबीन | Kharif | **11.9** | 18.4 | **64.8%** 🟠 | Madhya Pradesh (Soya Bowl), Maharashtra |
| **Maize** | मक्का | Kharif / All | **21.5** | 38.8 | **55.4%** 🟠 | Karnataka, MP, Bihar, Maharashtra |
| **Sorghum** | ज्वार (Shree Anna) | Kharif/Rabi | **10.3** | 19.7 | **52.2%** 🟠 | Maharashtra, Karnataka, Rajasthan, MP |

---

## 🧮 Domain Feature Engineering

The training pipeline constructs three engineered domain interaction features:

1. **Rainfall–Temperature Interaction:**
   $$\text{rain\_temp\_interaction} = \text{Rainfall (mm)} \times \text{Avg Temp (°C)}$$
   *Captures synergistic heatwave and drought stress thresholds.*

2. **Pesticide Application to Rainfall Ratio:**
   $$\text{pesticide\_per\_rainfall} = \frac{\text{Pesticides (tonnes)}}{\text{Rainfall (mm)} + 10^{-5}}$$
   *Models surface input wash-off and leaching during heavy downpours.*

3. **Years Since Base Year:**
   $$\text{years\_since\_start} = \text{Season Year} - 1990$$
   *Encodes longitudinal mechanization and Green Revolution technological trends.*

---

## 📁 Repository Structure

```
harvestline/
├── backend/
│   ├── app.py                      # Flask REST API with Indian analytics & scenario endpoints
│   ├── model/
│   │   ├── crop_yield_pipeline.pkl # Serialized Scikit-Learn Pipeline (RandomForest)
│   │   ├── lookup.json             # Country & crop historical baselines
│   │   └── trend.json              # 1990–2013 yearly harvest series
│   ├── templates/
│   │   └── index.html              # Modern Web Dashboard with Viva Guide & Mandi Units
│   └── static/
│       ├── style.css               # Design system, responsive cards & dark panels
│       ├── script.js               # Multi-unit converter, Chart.js graphs, presets & map
│       └── chart.umd.js            # Bundled Chart.js library
├── data/
│   └── yield_df.csv                # Clean training dataset (13,734 FAO records)
├── notebooks/
│   └── Harvestline_Crop_Yield_Prediction_ML_Pipeline.ipynb  # Interactive Colab Notebook
├── scripts/
│   ├── train_model.py              # CLI training pipeline script
│   └── generate_lookups.py         # JSON lookup generator from raw CSV
├── experiment_results.json         # Authoritative multi-model benchmark results & ablation data
├── run_experiments.py              # Experimental protocol & significance test runner
├── TEACHER_REVIEW_GUIDE.md         # Comprehensive viva voce & teacher defense cheatsheet
├── requirements.txt                # Global Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.10+
- `pip` & virtual environment

### 1. Clone & Navigate
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
Open **[http://localhost:5050](http://localhost:5050)** in your web browser.

---

## 🔌 API Reference

### `POST /api/predict`
Calculates yield prediction, confidence interval spread, and Explainable AI feature attribution.

**Request Payload:**
```json
{
  "area": "India",
  "item": "Wheat",
  "year": 2013,
  "rainfall": 1083.0,
  "temp": 26.0,
  "pesticides": 48459.0
}
```

**Response Output:**
```json
{
  "prediction_hg_ha": 24760.9,
  "prediction_t_ha": 2.476,
  "prediction_kg_ha": 2476.1,
  "band_t_ha": [2.201, 2.684],
  "delta_pct": -6.7,
  "xai_breakdown": [
    { "feature": "Regional Baseline Yield", "impact_t_ha": 2.476, "type": "base" },
    { "feature": "Temperature Adjustment", "impact_t_ha": 0.011, "type": "delta" },
    { "feature": "Rainfall Adjustment", "impact_t_ha": -0.034, "type": "delta" },
    { "feature": "Pesticide Use Adjustment", "impact_t_ha": 0.003, "type": "delta" },
    { "feature": "Climate Interaction Effect", "impact_t_ha": 0.020, "type": "delta" }
  ]
}
```

### `POST /api/scenarios_compare`
Simulates all 5 climate change scenarios simultaneously for side-by-side comparison graphing.

### `GET /api/india_insights`
Returns Indian crops, local Hindi names, seasonal cropping calendars (Kharif, Rabi, Zaid), agro-climatic zones, and Mandi unit conversion multipliers.

### `GET /api/evaluation`
Returns comprehensive multi-model benchmark statistics ($R^2$, RMSE, MAE, 5-Fold CV), feature ablation study (Table V), feature importance comparison (Impurity vs Permutation), paired bootstrap test results, and test scatter plot points.

---

## 👨‍🏫 Academic Review & Viva Defense Manual

For a complete guide containing:
- The **60-Second Elevator Pitch** for examiners
- **Step-by-step Architecture Flowchart**
- **Top 10 Teacher Viva Questions with Direct Answers** (Why RF? Why did SVR fail? What is Data Leakage? Why deduplicate?)
- **Glossary of Agronomic & ML Terms**

👉 **Read [TEACHER_REVIEW_GUIDE.md](TEACHER_REVIEW_GUIDE.md)**

---

## 📜 Dataset Source & License

Data sourced from public records compiled by the **Food and Agriculture Organization (FAO)** and **The World Bank** (1990–2013).

Distributed under the MIT License.
