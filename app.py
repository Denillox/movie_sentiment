import streamlit as st
import pickle
import re
import pandas as pd

# Function to clean the reviews, removing unnecessary text/symbols
def clean_review(text):
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = text.lower()
    words = text.split () 

    cleaned_text = " ".join(words)

    return cleaned_text

# Load modell and vectorizer
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


st.title("Movie Sentiment Analyzer")
st.write("Write a review down below and the AI will tell if it's positive or negative!")

user_input = st.text_area("Your review: ", placeholder="The movie was amazing...")

if st.button("Analyze"):
    if user_input.strip():
        cleaned = clean_review(user_input)
        
        vectorized_text = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized_text)

        probabilities = model.predict_proba(vectorized_text)[0]
        pos_score = probabilities[1]

        if prediction[0] == 1:
            st.success(f"This review is positive! (Confidence: {pos_score:.1%})")
        else:
            st.success(f"This review is negative! (Confidence: {probabilities[0]:.1%})")
    else:
        st.warning("Please write your review first.")

