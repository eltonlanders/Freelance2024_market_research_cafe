# -*- coding: utf-8 -*-
"""
@author: Elton Landers
"""


# src/topic_modeling.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def perform_lda(data, n_components=5):
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(data['cleaned_text'])

    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        topics.append([feature_names[i] for i in topic.argsort()[-10:]])

    return topics


