# -*- coding: utf-8 -*-
"""
@author: Elton Landers
"""

# src/feature_extraction.py
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_text(data):
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    return vectorizer.fit_transform(data)


