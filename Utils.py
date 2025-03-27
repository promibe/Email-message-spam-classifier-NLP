import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import joblib
import spacy
import cloudpickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the model and vectorizer

# Load the model
with open('voting_model.joblib', 'rb') as f:
    loaded_model = joblib.load(f)


# Data Cleaning Function
def clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    df = df.rename(columns={'v1': 'target', 'v2': 'text'})
    df = df[['target','text']]
    df['target'] = df['target'].map({'ham':0, 'spam':1})
    df = df.drop_duplicates()
    df.dropna(inplace=True)
    return df

url = "https://raw.githubusercontent.com/promibe/Email-message-spam-classifier-NLP/main/spam.csv"
df = clean_data(url)

nlp = spacy.load('en_core_web_sm')
#preprocessing function
def preprocess(text):
    doc = nlp(text)
    remaining_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or not token.text.isalnum(): #check whether a given token is stop word or punctuations, or special characters
            continue
        remaining_tokens.append(token.text.lower())
    return " ".join(remaining_tokens)

df['preprocessed_text'] = df['text'].apply(preprocess)

tfid = TfidfVectorizer(max_features=3000)

def vectorizer(df):
    tfid.fit_transform(df['preprocessed_text'])
    X_tfid = tfid.fit_transform(df['preprocessed_text']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X_tfid, df['target'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, tfid

X_train, X_test, y_train, y_test, loaded_vectorizer = vectorizer(df)

# Function to make predictions
def make_prediction(text):
    text = preprocess(text)
    trans_text = loaded_vectorizer.transform([text])
    
    # converting to array:
    X_test_dense = trans_text.toarray()
    prediction = loaded_model.predict(X_test_dense)[0]
    
    return prediction
