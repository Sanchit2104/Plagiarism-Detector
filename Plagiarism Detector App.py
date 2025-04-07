import streamlit as st
import pickle
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
stop_words=nltk.corpus.stopwords.words('english')
import numpy as np
model=pickle.load(open('model.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
st.title("Plagiarism Detector")
ip=st.text_input("Enter message: ")
vectorized_text = tfidf_vectorizer.transform([ip])
result = model.predict(vectorized_text)
if st.button("Detect"):
    if result[0]==0:
        final='No Plagiarism Detected'
    else:
        final='Plagiarism Detected'
    st.subheader(str(final))
