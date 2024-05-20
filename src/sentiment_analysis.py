# -*- coding: utf-8 -*-
"""
@author: Elton Landers
"""

# src/sentiment_analysis.py
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_aspect_sentiment(aspects, text):
    results = {}
    for aspect in aspects:
        if aspect in text:
            sentiment = sentiment_analyzer(aspect)
            results[aspect] = sentiment[0]['label']
    return results


