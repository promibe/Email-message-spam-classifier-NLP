import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Load the model and vectorizer
loaded_model = pickle.load(open("spam_detector_model.pkl", "rb"))
loaded_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

#preprocessing function
def preprocess(text):
    doc = nlp(text)
    remaining_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or not token.text.isalnum(): #check whether a given token is stop word or punctuations, or special characters
            continue
        remaining_tokens.append(token.text.lower())
    return " ".join(remaining_tokens)

# Function to make predictions
def make_prediction(text):
    text = preprocess(text)
    trans_text = loaded_vectorizer.transform([text])
    
    # converting to array:
    X_test_dense = trans_text.toarray()
    prediction = loaded_model.predict(X_test_dense)[0]
    
    return prediction