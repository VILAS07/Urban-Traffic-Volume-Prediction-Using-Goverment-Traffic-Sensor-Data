# 🚦 Urban Traffic Volume Prediction using Machine Learning

## 📌 Project Overview

This project develops an end-to-end Machine Learning system to predict traffic volume using historical road sensor data.

Using **273,000+ real traffic observations**, the model learns patterns such as peak hours, vehicle types, road locations, and long-term traffic growth trends to accurately estimate vehicle counts.

The solution can support **traffic planning, congestion control, and smart city infrastructure decisions.**

---

## 🎯 Objective

To build a robust regression model capable of predicting the number of vehicles passing a traffic sensor based on temporal and road characteristics.

---

## 📊 Dataset

* Source: Government Traffic Sensor Dataset
* Records: 273,913
* Initial Features: 27
* Final Features after preprocessing & engineering: 15

Dataset includes:

* Traffic count (target variable)
* Station location
* Time period (Peak, Weekend, Holiday)
* Vehicle classification
* Traffic direction
* Year and temporal indicators

---

## 🧹 Data Preprocessing

Key preprocessing steps performed:

* Removed columns with high missing values or no predictive importance
* Corrected sensor error values (-1 replaced using median imputation)
* Filtered invalid traffic counts (≤0)
* Removed incomplete year data (2026)
* Eliminated redundant categorical sequence columns

These steps ensured **clean, reliable input data for model training.**

---

## ⚙️ Feature Engineering

To improve model performance, several meaningful features were created:

* **Peak Hour Indicator** → identifies rush-hour traffic patterns
* **Weekend Indicator** → captures behavioural traffic differences
* **Holiday Indicator** → models unusual traffic fluctuations
* **Both Direction Indicator** → accounts for combined traffic counts
* **Heavy Vehicle Indicator** → distinguishes freight vs passenger traffic
* **Decade Feature** → captures long-term growth trends

Feature engineering significantly enhanced the model’s ability to learn **real-world traffic behaviour.**

---

## 📊 Exploratory Data Analysis

EDA revealed important insights:

* Traffic distribution is right-skewed with few extremely busy roads
* Overall traffic volume shows long-term growth trends
* Peak periods exhibit significantly higher vehicle counts
* Heavy vehicle roads show comparatively lower counts
* COVID-19 period caused noticeable traffic decline

Multiple visualizations such as distribution plots, trend analysis, correlation heatmaps, and boxplots were used.

---

## 🤖 Model Development

Steps followed:

1. Train-test split (80% training, 20% testing)
2. Log transformation of target variable to handle skewness
3. Training and evaluation of multiple regression models including:

   * Linear Regression
   * Decision Tree
   * Random Forest
   * Gradient Boosting
   * KNN
   * SVR
   * Extra Trees

---

## 🏆 Best Model Performance

**Random Forest Regressor achieved the best results:**

* R² Score: **0.9856**
* Mean Absolute Error: **≈ 537 vehicles**
* Strong predictive stability across different traffic scenarios

---

## ⭐ Key Findings

* Station location is the strongest determinant of traffic volume
* Rush hours significantly increase vehicle flow
* Heavy vehicle routes show different traffic behaviour
* Traffic trends evolve across decades due to urban growth

---

## 🚀 Project Execution

Clone repository:

git clone https://github.com/VILAS07/Urban-Traffic-Volume-Prediction-Using-Goverment-Traffic-Sensor-Data.git

Install dependencies:

pip install -r requirements.txt

Run Streamlit application:

streamlit run app.py

---

## 🔮 Future Improvements

* Deep Learning models (LSTM / Temporal Models)
* Real-time traffic prediction API
* Deployment using cloud platforms
* Integration with weather and GPS data
* Live traffic dashboard

---

## 👨‍💻 Author

**Vilas PK**
B.Tech Artificial Intelligence & Data Science
Machine Learning & Data Science Enthusiast
