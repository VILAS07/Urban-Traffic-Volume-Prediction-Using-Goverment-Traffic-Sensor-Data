# ════════════════════════════════════════════════
# README 1 — Urban Traffic Volume Prediction
# Paste into: Urban-Traffic-Volume-Prediction repo
# ════════════════════════════════════════════════
 
# Urban Traffic Volume Prediction
 
Predicts road traffic volume using 273,913 rows of real NSW government traffic sensor data. Built an end-to-end ML pipeline with feature engineering, model comparison, and a live Streamlit app.
 
## Results
| Metric | Value |
|--------|-------|
| Best Model | Random Forest |
| R² Score | **0.9856** |
| Mean Prediction Error | ~537 vehicles |
| Models Compared | 9 |
 
## What I Built
- Cleaned 273,913 rows — removed 13 redundant columns, fixed -1 sensor errors, dropped incomplete years
- Engineered 6 new features: `is_peak`, `is_weekend`, `is_holiday`, `is_heavy`, `is_both_directions`, `decade`
- Applied log transformation on target variable to handle right-skewed distribution
- Compared 9 ML models using Scikit-learn Pipelines with label encoding
- Deployed interactive prediction app via Streamlit
 
## Tech Stack
`Python` `Scikit-learn` `Random Forest` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Streamlit`
 
## Key Finding
Station ID (road location) was the #1 predictor of traffic volume — more important than time of day or vehicle type.
 
## Run Locally
```bash
git clone https://github.com/VILAS07/Urban-Traffic-Volume-Prediction-Using-Goverment-Traffic-Sensor-Data
pip install -r requirements.txt
streamlit run app.py
```
 
---
