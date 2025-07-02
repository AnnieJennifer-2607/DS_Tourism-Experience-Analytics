# 🧭 Tourism Intelligence Suite | ML-Based Rating Prediction, Visit Mode Classification & Personalized Recommendations

A comprehensive machine learning project to transform tourism experiences through predictive analytics and intelligent recommendations.

---

## 🔍 Overview

This project tackles key challenges faced by tourism platforms through:
- **🎯 Regression**: Predicting user ratings for tourist attractions.
- **🧳 Classification**: Identifying visit modes (Family, Business, etc.).
- **🌍 Recommendation**: Suggesting attractions based on user preferences.

It supports personalized travel planning, customer segmentation, and tourism analytics, enhancing retention and user satisfaction.

---

## 💼 Business Applications

- Recommend attractions tailored to user profiles.
- Identify top-rated regions and travel trends.
- Segment travelers for targeted marketing.
- Improve service by anticipating visitor behavior.

---

## 🧠 ML Pipeline

### 📁 Data Processing
- Cleaned missing values & standardized formats.
- Merged user, attraction & transaction data.
- Encoded categorical features (OneHot, Target).
- Scaled numeric features for model performance.

### 📊 EDA Highlights
- Explored trends in ratings, attraction popularity.
- Analyzed visit modes by demographics & region.
- Uncovered patterns for segmentation & targeting.

### 🤖 Modeling
- **Regression**: XGBoost for predicting attraction ratings.
- **Classification**: Random Forest/XGBoost for visit mode.
- **Recommendation**: Collaborative filtering (user-item matrix) & content-based filtering.

### 📈 Evaluation
- **Regression**: R² Score, MSE.
- **Classification**: Accuracy, F1, Recall.
- **Recommendation**: RMSE, MAP.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Surprise
- **Web App**: Streamlit
- **Model Serialization**: Joblib
- **Visualization**: Matplotlib, Seaborn, Plotly

---

## 🚀 Streamlit App Features

- 🧾 Inputs: User location, attraction type, visit history.
- 📊 Outputs:
  - Rating prediction
  - Visit mode classification
  - Top personalized attraction recommendations
- 🎨 UI: Travel-themed with Lottie animations, banner, and analytics dashboard.
- 📍 Visuals: Popular attractions, regions, and behavior breakdown.
