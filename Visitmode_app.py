import streamlit as st
import pandas as pd
import joblib

# Load models and encoders
best_xgb = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\xgboost_model(class).pkl")
ohe = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\onehot_encoder(class).pkl")
target_enc_attraction = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\target_enc_attraction(class).pkl")
target_enc_country = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\target_enc_country(class).pkl")
label_encoder = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\label_encoder(class).pkl")

# Load expected feature columns list
expected_features = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\feature_columns.pkl")

# Load dataset for select box options
df = pd.read_csv(r"C:\Users\annie\DS\VScode\Capstone Project 4\Dataset_Tourism_final.csv")

st.set_page_config(page_title="Visit Mode Predictor", layout="wide")

st.markdown("""
    <style>
    .banner {
        background-color: #1f4e79;
        color: white;
        padding: 25px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        border-radius: 0 0 15px 15px;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }
    .Visitmode-box {
        background-color: #f0f8ff;
        border-left: 5px solid #4682B4;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 500;
        color: #333;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4682B4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.4em;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='banner'>🚗 Predict Your Ideal Visit Mode</div>", unsafe_allow_html=True)

st.sidebar.image(r"C:\Users\annie\DS\VScode\Capstone Project 4\tourismrect.png", width=300)
st.sidebar.markdown("### ✨ Travel Smart")
st.sidebar.success("“The journey matters more than the destination.”")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🧠 **How It Works**")
st.sidebar.markdown("""
1️⃣ Fill in your **travel preferences**  
2️⃣ Click **Predict** to get visit mode  
3️⃣ Explore your **ideal travel style**  
""")
st.sidebar.info("Built with machine learning 💡")

st.markdown("## 🎯 Visit Mode Prediction System")
st.write("Fill in your travel details and predict how the user might visit a destination!")

selected_features = ["VisitYear", "VisitMonth", "ContinentId", "Country", "Attraction", "AttractionType"]

col1, col2 = st.columns(2)

with col1:
    visit_year = st.selectbox("📅 Visit Year", options=sorted(df["VisitYear"].unique()))
    visit_month = st.selectbox("📆 Visit Month", options=sorted(df["VisitMonth"].unique()))
    country = st.selectbox("🌎 Country", options=sorted(df["Country"].unique()))

with col2:
    continent_id = st.selectbox("🗺️ Continent ID", options=sorted(df["ContinentId"].unique()))
    attraction = st.selectbox("🎡 Attraction Name", options=sorted(df["Attraction"].unique()))
    attraction_type = st.selectbox("🎭 Attraction Type", options=sorted(df["AttractionType"].unique()))

if st.button("🔮 Predict Visit Mode"):
    try:
        input_data = pd.DataFrame([[visit_year, visit_month, continent_id, country, attraction, attraction_type]],
                                  columns=selected_features)

        # Target Encoding
        input_data["Attraction"] = target_enc_attraction.transform(input_data[["Attraction"]])
        input_data["Country"] = target_enc_country.transform(input_data[["Country"]])

        # One-hot Encoding
        encoded_features = ohe.transform(input_data[["AttractionType"]])
        encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(["AttractionType"]))
        
        # Add missing columns if any and reorder to match training
        for col in expected_features:
            if col not in input_data.columns and col not in encoded_df.columns:
                # Add missing columns to encoded_df with 0
                encoded_df[col] = 0
        
        # Reorder encoded_df columns to match those in expected_features (only those from encoded_df)
        encoded_cols = [col for col in expected_features if col in encoded_df.columns]
        encoded_df = encoded_df[encoded_cols]

        # Drop AttractionType column and concat encoded one-hot columns
        input_data = input_data.drop(columns=["AttractionType"])
        input_data = pd.concat([input_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Reindex to match expected features
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        # Predict
        prediction = best_xgb.predict(input_data)
        predicted_category = label_encoder.inverse_transform(prediction)[0]

        st.markdown(
            f"<div class='Visitmode-box'>🧭 Predicted Visit Mode: <strong>{predicted_category}</strong></div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"🚨 Prediction failed: {str(e)}")
        st.write("Debug info:")
        st.write(input_data)

# # Dataset preview
# st.markdown("### 📊 Preview of Tourism Data")
# st.write("Quick look at the dataset used to train the model:")
# st.dataframe(df.head(100))
