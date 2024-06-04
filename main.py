# -*- coding: utf-8 -*-
"""
@author: Elton Landers
"""


# main.py
import pandas as pd
from src.data_processing import load_data, handle_missing_values, preprocess_content
from src.nlp_utils import extract_entities, filter_relevant_entities, tokenize_content, stem_tokens, lemmatize_tokens
from src.sentiment_analysis import analyze_aspect_sentiment
from src.topic_modeling import perform_lda
from src.wordcloud_generation import generate_wordcloud
from src.feature_extraction import vectorize_text
from src.model_training import train_bert_model

# Load the data
data = load_data('data/data.csv')

# Preprocess the data
data = handle_missing_values(data)
data = preprocess_content(data)

# Tokenize, stem, and lemmatize
data['content_tokens'] = data['content'].apply(tokenize_content)
data['content_stemmed'] = data['content_tokens'].apply(stem_tokens)
data['content_lemmatized'] = data['content_tokens'].apply(lemmatize_tokens)

# Extract entities and filter them
data['entities'] = data['content'].apply(extract_entities)
data['filtered_entities'] = data['entities'].apply(filter_relevant_entities)

# Perform topic modeling (LDA)
topics = perform_lda(data)

# Generate a word cloud for the cuisines
cuisine_text = ' '.join(data['cuisine_mentions'].sum())
generate_wordcloud(cuisine_text)

# Example of aspect-based sentiment analysis
data['aspect_sentiments'] = data.apply(lambda row: analyze_aspect_sentiment(row['cuisine_mentions'] + row['theme_mentions'], row['content']), axis=1)

# Train BERT model (example with existing processed data)
# For this, ensure you have processed data (sentiment) in a compatible format
# model, history = train_bert_model(train_dataset, val_dataset)

# Save processed data
data.to_csv("processed.csv", index=False)

