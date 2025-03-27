import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from Utils import make_prediction, preprocess
import joblib
import spacy
import cloudpickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the Streamlit app UI
st.title("📨 Spam Detector")
st.write("Enter a message below to check if it's spam or not:")

# Get user input
text = st.text_area("Message", height=200)

# Predict when button is clicked
if st.button("Check"):
    if not text.strip():
        st.warning("Please enter a message to check.")
    else:
        prediction = make_prediction(text)
        if prediction == 0:
            st.success("✅ This message is **NOT SPAM**.")
        else:
            st.error("🚨 This message is **SPAM**.")

st.markdown("---")
st.caption("Built with ❤️ by Promise Ibediogwu Ekele")
