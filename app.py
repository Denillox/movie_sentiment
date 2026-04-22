import streamlit as st
import pickle
import re
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

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
    # Ord som vi aldrig vill highlighta eftersom de inte bär på sentiment
    ignore_list = [
        "is", "was", "the", "a", "an", "and", "it", "to", "of", "i", 
        "in", "with", "for", "as", "at", "by", "this", "that", "there"
    ]
    
    words = text.split()
    i = 0
    highlighted_html = ""
    
    while i < len(words):
        # Tvätta orden för logik-matchning
        current_clean = re.sub(r'[^a-z]', '', words[i].lower())
        
        # Kolla Bigram först
        bigram = ""
        if i < len(words) - 1:
            next_clean = re.sub(r'[^a-z]', '', words[i+1].lower())
            bigram = f"{current_clean} {next_clean}"
        
        # 1. Kolla om vi har ett meningsfullt Bigram (t.ex. "not great")
        # Vi kollar inte ignore_list här, för "not" + "great" är viktigt tillsammans!
        if bigram in word_weights and abs(word_weights[bigram]) > 1.0:
            weight = word_weights[bigram]
            color = "#f8d7da" if weight < 0 else "#d4edda"
            highlighted_html += f'<span style="background-color: {color}; color: black; padding: 2px 5px; border-radius: 3px; border: 1px solid; font-weight: bold;">{words[i]} {words[i+1]}</span> '
            i += 2
            continue
        
        # 2. Kolla enskilt ord (OM det inte finns i ignore_list)
        weight = word_weights.get(current_clean, 0)
        
        if current_clean not in ignore_list and abs(weight) > 1.2:
            color = "#d4edda" if weight > 0 else "#f8d7da"
            highlighted_html += f'<span style="background-color: {color}; color: black; padding: 2px 5px; border-radius: 3px; border: 1px solid; font-weight: bold;">{words[i]}</span> '
        else:
            highlighted_html += f"{words[i]} "
        
        i += 1
            
    return highlighted_html

# Load model and vectorizer
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/word_weights.json', 'rb') as f:
    word_weights = json.load(f)

st.set_page_config(
    page_title="CinemaSentiment AI",
    page_icon="🎬",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1f1f1f;
        border-right: 1px solid #333;
    }
    
    .stTextArea textarea {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
    }
    
    .stButton>button {
        background-color: #e50914 !important; /* Netflix röd */
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        padding: 0.6rem 2rem !important;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #f40612 !important;
        transform: scale(1.02);
    }

    .stAlert {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    h1, h2, h3 {
        color: #e50914 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.title("Settings")
    page = st.radio("Choose Mode:", ["Single Review Analysis", "Multiple Review Analysis", "Model Insights"])
    st.info("This app uses a Logistic Regression model trained on 50,000 IMDB reviews.")

# For the user to input a single review and get a result from the AI
if page == "Single Review Analysis":
    st.header("Manual Sentiment Analysis")
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

            st.write("----")
            col1, col2 = st.columns([1, 2])
    
            with col1:
                if pos_score > 0.6:
                    st.markdown("### Result: 🟩 Positive")
                elif pos_score < 0.4:
                    st.markdown("### Result: 🟥 Negative")
                else:
                    st.markdown("### Result: 🟨 Neutral")
                
                st.metric("Confidence Score", f"{pos_score:.1%}")

            with col2:
                st.write("### Positivity Meter")
                st.progress(pos_score)
                if pos_score > 0.8:
                    st.caption("The AI is very confident this is a glowing review!")
        else:
            st.warning("Please write your review first.")

# For the user to input multiple reviews at the same time
elif page == "Multiple Review Analysis":
    st.header("Bulk Sentiment Analysis")
    st.write("Copy and paste multiple reviews below. The AI will analyze each paragraph and show you the overall mood.")

    user_paste = st.text_area("Paste reviews here (separate with new lines):", 
                              height=300, 
                              placeholder="Review 1: This movie was great!\n\nReview 2: I didn't like it at all...")

    if st.button("Analyze All"):
        if user_paste.strip():
            review_list = [line.strip() for line in user_paste.split('\n') if len(line.strip()) > 10]

            if not review_list:
                st.warning("Please paste some actual text to analyze.")
            else:
                st.info(f"Analyzing {len(review_list)} segments...")

                results = {"Positive": 0, "Negative": 0, "Neutral": 0}

                for r in review_list:
                    cleaned = clean_review(r)
                    vec = vectorizer.transform([cleaned])
                    prob = model.predict_proba(vec)[0][1]
                    
                    if 0.45 <= prob <= 0.55:
                        results["Neutral"] += 1
                    elif prob > 0.55:
                        results["Positive"] += 1
                    else:
                        results["Negative"] += 1

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("### Summary Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Positive", results["Positive"])
                    m2.metric("Negative", results["Negative"])
                    m3.metric("Neutral", results["Neutral"])

                with col2:
                    fig, ax = plt.subplots()
                    colors = ['#d4edda', '#f8d7da', '#fff3cd'] # Grön, Röd, Gul
                    ax.pie(results.values(), labels=results.keys(), autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)

                with st.expander("See detailed breakdown"):
                    for r in review_list:
                        cleaned = clean_review(r)
                        vec = vectorizer.transform([cleaned])
                        score = model.predict_proba(vec)[0][1]
                        
                        if score > 0.55:
                            label = "Positive"
                        elif score < 0.45:
                            label = "Negative"
                        else:
                            label = "Neutral"
                        
                        st.write(f"{label} | {r[:100]}...")
        else:
            st.warning("The text box is empty!")

# Info/statistics about the model used
elif page == "Model Insights":
    st.header("Model Performance & Insights")
    st.write("Below it is showed how the model was evaluated and what it actually learned.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "88.92%")
    col2.metric("Training Size", "35,000")
    col3.metric("Test Size", "15,000")

    st.subheader("Confusion Matrix")
    st.write("This matrix shows how many times the model guessed right vs. wrong on the test data.")

    cm_data = pd.DataFrame(
        [[6483, 928], [734, 6855]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )
    st.table(cm_data)

    st.subheader("Top Predictive Words")
    st.write("These are the words that the model associates most strongly with each sentiment.")

    sorted_weights = sorted(word_weights.items(), key=lambda item: item[1])
    neg_features = sorted_weights[:10]
    pos_features = sorted_weights[-10:]
    pos_features.reverse()

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.write("Top Negative")
        for word, weight in neg_features:
            st.write(f"{word}: {weight:.2f}")
    
    with f_col2:
        st.write("Top Positive")
        for word, weight in pos_features:
            st.write(f"{word}: {weight:.2f}")

