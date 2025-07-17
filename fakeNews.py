import streamlit as st
import joblib
import requests
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

# Load ML model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector with Real-Time News")

# --- User Input
user_input = st.text_area("🔍 Enter news headline or article below:")

if st.button("Check Manually"):
    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)
        prob = model.predict_proba(vec_input)

        label = prediction[0]
        confidence = np.max(prob) * 100

        if label == "FAKE":
            st.error(f"🚫 This news is **FAKE** with {confidence:.2f}% confidence.")
        else:
            st.success(f"✅ This news is **REAL** with {confidence:.2f}% confidence.")

# --- Real-Time News Checker
st.markdown("---")
st.subheader("🌍 Check Real-Time News")

gnews_api_key = "7faf0c862750519f76a559a21c99c3f2"  # Replace with your API key

if st.button("Fetch & Analyze Top Headlines"):
    st.info("⏳ Fetching the latest news...")
    url = f"https://gnews.io/api/v4/top-headlines?country=in&lang=en&token={gnews_api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])

        # Shuffle and pick any 5 random headlines
        random.shuffle(articles)
        articles = articles[:5]

        for i, article in enumerate(articles):
            title = article.get("title", "No title")
            st.markdown(f"### 📰 Headline {i+1}: {title}")
            vec_title = vectorizer.transform([title])
            pred = model.predict(vec_title)[0]
            prob = model.predict_proba(vec_title)
            conf = np.max(prob) * 100

            if pred == "FAKE":
                st.error(f"🛑 Prediction: **FAKE** ({conf:.2f}% confidence)")
            else:
                st.success(f"✅ Prediction: **REAL** ({conf:.2f}% confidence)")
    else:
        st.error("❌ Failed to fetch news from GNews. Please check your API key or internet.")
