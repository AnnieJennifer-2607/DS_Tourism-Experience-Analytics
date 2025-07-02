import streamlit as st
import joblib
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

# --- Load ML models and encoders ---
model_path = r"C:\Users\annie\DS\VScode\Capstone Project 4\XGBoost model(regression).pkl"
ohe_path = r"C:\Users\annie\DS\VScode\Capstone Project 4\One-Hot Endcoder Model(regression).pkl"
target_enc_path = r"C:\Users\annie\DS\VScode\Capstone Project 4\Target Encoder Model(regression).pkl"
scaler_path = r"C:\Users\annie\DS\VScode\Capstone Project 4\Scaler(regression).pkl"

best_xgb = joblib.load(model_path)
ohe = joblib.load(ohe_path)
target_enc = joblib.load(target_enc_path)
scaler = joblib.load(scaler_path)

# --- Load dataset ---
df_path = r"C:\Users\annie\DS\VScode\Capstone Project 4\Dataset_Tourism_final.csv"
df = pd.read_csv(df_path)

country_mapping = dict(zip(df["Country"], df["CountryId"]))
region_mapping = dict(zip(df["Region"], df["RegionId"]))

# --- CSS Styling ---
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.98);
        padding: 20px;
        border-radius: 12px;
    }
    .custom-header {
        font-size: 38px;
        color: #1E3D59;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        margin-top: 10px;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #104E8B;
        color: #fff;
    }
    .stSuccess {
        background-color: #e0f7fa;
        border-left: 6px solid #0097a7;
        padding: 10px;
        border-radius: 8px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image(r"C:\Users\annie\DS\VScode\Capstone Project 4\tourism_banner_bg.jpg", width=300)
st.sidebar.markdown("## ✈️ Welcome Explorer!")
st.sidebar.markdown("""
> _"Travel far enough, you meet yourself."_  
> — David Mitchell

---

🔍 **Rate Your Trip Before You Take It!**  
🎯 Instant predictions using AI magic.

---

### 📚 Did You Know?
- 🧳 Tourism boosts local economies  
- 🗺️ The Eiffel Tower can grow 6 inches in summer  
- ✈️ Travel enhances creativity
""")

# --- Page Header Image ---
st.markdown("<br>", unsafe_allow_html=True)
st.image("tourismrect.png", caption="Explore the World with AI-Powered Predictions", use_column_width=True)

# --- Main Page Header ---
st.markdown('<div class="custom-header">🏝️ Tourism Rating Prediction App</div>', unsafe_allow_html=True)
st.markdown("### 📋 Enter your trip details:")

# --- Input Fields in Pairs ---
# Row 1: Country & Region
col1, col2 = st.columns(2)
with col1:
    selected_country = st.selectbox("🌍 Select Country", options=sorted(df["Country"].unique()))
with col2:
    filtered_regions = df[df["Country"] == selected_country]["Region"].unique()
    selected_region = st.selectbox("📍 Select Region", options=sorted(filtered_regions))

CountryId = country_mapping[selected_country]
RegionId = region_mapping[selected_region]

# Filter attractions
filtered_attractions = df[(df["Country"] == selected_country) & (df["Region"] == selected_region)]
attraction_options = filtered_attractions[["AttractionId", "Attraction", "AttractionType"]].drop_duplicates()

# Row 2: Attraction ID & Name
col3, col4 = st.columns(2)
with col3:
    selected_attraction_id = st.selectbox("🎡 Attraction ID", options=sorted(attraction_options["AttractionId"].unique()))
with col4:
    selected_attraction_name = st.selectbox("🏞️ Attraction Name", options=sorted(attraction_options["Attraction"].unique()))

# Row 3: Attraction Type & Visit Mode
col5, col6 = st.columns(2)
with col5:
    selected_attraction_type = st.selectbox("🏛️ Attraction Type", options=sorted(attraction_options["AttractionType"].unique()))
with col6:
    VisitModeName = st.selectbox("🚗 Visit Mode", options=sorted(df["VisitModeName"].unique()))

# Row 4: Year & Month
col7, col8 = st.columns(2)
with col7:
    VisitYear = st.number_input("📅 Visit Year", min_value=2000, max_value=2100, step=1)
with col8:
    VisitMonth = st.number_input("📆 Visit Month", min_value=1, max_value=12, step=1)

# --- Predict Button ---
if st.button("🔍 Predict Rating"):
    input_data = pd.DataFrame({
        "VisitYear": [VisitYear],
        "VisitMonth": [VisitMonth],
        "VisitModeName": [VisitModeName],
        "AttractionId": [selected_attraction_id],
        "Attraction": [selected_attraction_name],
        "AttractionType": [selected_attraction_type],
        "CountryId": [CountryId],
        "RegionId": [RegionId]
    })

    encoded_features = ohe.transform(input_data[["VisitModeName", "AttractionType"]])
    encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(["VisitModeName", "AttractionType"]))

    input_data["Attraction"] = target_enc.transform(input_data["Attraction"])
    input_data.drop(columns=["VisitModeName", "AttractionType"], inplace=True)
    input_data = pd.concat([input_data, encoded_df], axis=1)

    input_scaled = scaler.transform(input_data)
    prediction = best_xgb.predict(input_scaled)

    st.success(f"⭐ **Predicted Tourism Rating:** {prediction[0]:.2f}")
