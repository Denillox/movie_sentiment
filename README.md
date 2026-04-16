Small Project which shows the process of transforming raw data into a funciton web app using streamlit that analyzes the sentiment of movie reviews (dataset from kaggle, can be downloaded here: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
The reviews are cleaned of HTML tags and noice, then converted into numerical values allowing the model to prioritize meaningful words over common stop words.

The model used is a Logistic Regression trained on 50 000 IMDb reviews to identify patterns associated with positive and negative feedback. Everything is wrapped in a simple interface built with streamlit (for now), where users can input any text and receive and immediate sentiment analysis.

Install everything needed by running "pip install -r requirements.txt"
