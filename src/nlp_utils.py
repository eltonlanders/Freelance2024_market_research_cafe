# -*- coding: utf-8 -*-
"""
@author: Elton Landers
"""

# src/nlp_utils.py
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def filter_relevant_entities(entities):
    relevant_labels = ['FOOD', 'GPE', 'ORG', 'PRODUCT']
    return [ent for ent in entities if ent[1] in relevant_labels]

def tokenize_content(content):
    return word_tokenize(content)

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]




