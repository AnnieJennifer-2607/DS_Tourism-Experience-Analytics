import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

# Load the tourism dataset
Tourism_df = pd.read_csv(r"C:\Users\annie\DS\VScode\Capstone Project 4\Dataset_Tourism_final.csv")
id_to_name = pd.Series(Tourism_df.Attraction.values, index=Tourism_df.AttractionId).to_dict()

# Page config
st.set_page_config(page_title="🌏 Tourism Recommender", layout="wide")

# ---------- 💠 STYLING ----------
st.markdown("""
    <style>
    /* Banner */
    .banner {
        background-color: #003366;
        padding: 25px;
        text-align: center;
        border-radius: 0 0 12px 12px;
        color: white;
        font-size: 28px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 30px;
    }
    /* Recommendation box */
    .recommendation-box {
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
    /* Button */
    .stButton>button {
        background-color: #4682B4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        margin-top: 10px;
    }
    /* Header */
    h3 {
        color: #2F4F4F;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- 🌄 BANNER ----------
st.markdown("<div class='banner'>🌍 Discover Your Next Adventure</div>", unsafe_allow_html=True)

# ---------- 🧠 SIDEBAR ----------
quotes = [
    "“Travel is the only thing you buy that makes you richer.”",
    "“Jobs fill your pocket, but adventures fill your soul.”",
    "“Life is short and the world is wide.”",
    "“Collect moments, not things.”",
    "“To travel is to live.”",
    "“Wander often, wonder always.”"
]
quote_choice = random.choice(quotes)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=300)
st.sidebar.markdown("### ✈️ *Wanderlust Vibes*")
st.sidebar.info(quote_choice)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ℹ️ **How it Works**")
st.sidebar.markdown("""
1️⃣ Select a **User ID**  
2️⃣ Click on **'Get Recommendations'**  
3️⃣ Explore your **Top 5 Destination Picks**  
""")
st.sidebar.success("AI-Powered Recommendations!")

# ---------- 🔌 Load Models ----------
svd = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\svd(recommend).plk")
knn_model = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\KNN(recommend).plk")
user_attraction_matrix = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\use attraction matrix(recommend).plk")
user_attraction_matrix_reduced = joblib.load(r"C:\Users\annie\DS\VScode\Capstone Project 4\user attraction matrix reduce(recommend).plk")

# ---------- 🎯 User Input ----------
st.markdown("<h3>Select User ID for Personalized Recommendations</h3>", unsafe_allow_html=True)
user_ids = user_attraction_matrix.index.tolist()
user_id = st.selectbox("🔍 Choose a User ID", user_ids)

# ---------- 🚀 Recommendation Logic ----------
def recommend_attractions(user_id, num_recommendations=5):
    if user_id not in user_attraction_matrix.index:
        return "User ID not found! Try with a different ID."

    user_idx = user_attraction_matrix.index.get_loc(user_id)
    distances, indices = knn_model.kneighbors([user_attraction_matrix_reduced[user_idx]], n_neighbors=7)
    similar_users = user_attraction_matrix.index[indices.flatten()[1:]]

    user_ratings = user_attraction_matrix.loc[user_id]
    unseen_attractions = user_ratings[user_ratings == 0].index

    attraction_scores = {}
    for sim_user, dist in zip(similar_users, distances.flatten()[1:]):
        for attraction in unseen_attractions:
            attraction_scores[attraction] = attraction_scores.get(attraction, 0) + (
                user_attraction_matrix.loc[sim_user, attraction] * (1 - dist)
            )

    recommended_attractions = sorted(attraction_scores, key=attraction_scores.get, reverse=True)[:num_recommendations]
    return recommended_attractions if recommended_attractions else "No new recommendations found."

# ---------- 🎁 Show Recommendations ----------
if st.button("🎒 Get Recommendations"):
    recommended = recommend_attractions(user_id)
    st.markdown("<h3>🎯 Recommended Attractions</h3>", unsafe_allow_html=True)

    if isinstance(recommended, list):
        for i, attraction_id in enumerate(recommended, 1):
            attraction_name = id_to_name.get(attraction_id, "Unknown Name")
            st.markdown(
                f"<div class='recommendation-box'>✨ <b>{i}. {attraction_name}</b> (ID: {attraction_id})</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning(recommended)

# # ---------- 📊 Dataset Preview ----------
# st.markdown("### 📂 Preview of Tourism Data")
# st.write("Here's a quick peek into the dataset used for recommendation:")
# st.dataframe(Tourism_df)
