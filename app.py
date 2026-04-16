import streamlit as st
import pickle
import re
import pandas as pd
import json

# Function to clean the reviews, removing unnecessary text/symbols
def clean_review(text):
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = text.lower()
    words = text.split () 

    cleaned_text = " ".join(words)

    return cleaned_text

# Function to highlight positive and negative words in red/green
def highlight_text(text):
    words = text.split()
    highlighted_html = "" 

    for word in words:
        clean_word = word.lower().strip(".,!?:;\"")
        weight = word_weights.get(clean_word, 0)

        
        if weight > 1.5: 
            color = "#d4edda"
            highlighted_html += f'<span style="background-color: {color}; color: black; padding: 2px 5px; border-radius: 3px; border: 1px solid #c3e6cb; font-weight: bold;">{word}</span> '
        elif weight < -1.5:  
            color = "#f8d7da"
            highlighted_html += f'<span style="background-color: {color}; color: black; padding: 2px 5px; border-radius: 3px; border: 1px solid #f5c6cb; font-weight: bold;">{word}</span> '
        else:
            highlighted_html += f"{word} "
    
    return highlighted_html


# Load model and vectorizer
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/word_weights.json', 'rb') as f:
    word_weights = json.load(f)



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

        st.markdown('Feature Analysis: (Green means the word had positive impact, red means negative) ')
        st.markdown(highlight_text(user_input), unsafe_allow_html=True)
        st.info("Note: Highlights show individual word importance. The AI also considers word combinations (like 'not great') for its final decision.")

        if 0.4 <= pos_score <= 0.6:
            st.warning(f"This review feels a bit mixed, not sure if positive or negative. (Confidence: {pos_score:.2%})")
        elif pos_score > 0.6:
            st.success(f"This review is positive! (Confidence: {pos_score:.1%})")
        else:
            st.success(f"This review is negative! (Confidence: {probabilities[0]:.1%})")
    else:
        st.warning("Please write your review first.")

