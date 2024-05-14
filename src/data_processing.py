# -*- coding: utf-8 -*-
"""
@author: Elton Landers
"""

# src/data_processing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(data):
    missing_count = data['rating'].isnull().sum()
    missing_percentage = (missing_count / len(data)) * 100
    print(f"Missing values in 'rating': {missing_count} ({missing_percentage:.2f}%)")
    rating_median = data['rating'].median()
    data['rating'] = data['rating'].fillna(rating_median)
    return data

def preprocess_content(data):
    stop_words = set(stopwords.words('english'))
    data['content'] = data['content'].str.lower()
    data['content'] = data['content'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    data['content'] = data['content'].apply(lambda x: re.sub(r'[^\w\s,]', '', x))
    data['content'] = data['content'].apply(lambda x: re.sub(r'<.*?>', '', x))  # Remove HTML tags
    data['content'] = data['content'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))  # Remove URLs
    data['content'] = data['content'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())  # Remove extra whitespace

    data['content'] = data['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return data
