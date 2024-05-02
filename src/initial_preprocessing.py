# -*- coding: utf-8 -*-
"""
@author: Elton Landers
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import spacy 


data = pd.read_csv('data.csv')


missing_count = data['rating'].isnull().sum()
missing_percentage = (missing_count / len(data)) * 100
print(f"Missing values in 'rating': {missing_count} ({missing_percentage:.2f}%)")



rating_median = data['rating'].median()
data['rating'] = data['rating'].fillna(rating_median)


qualitative_data = data[(data['sentiment'].notnull()) & (data['content'].notnull())]

missing_after = data['rating'].isnull().sum()


data['content'] = qualitative_data['content'].str.lower() 

data['content'] = data['content'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

data['content'] = data['content'].apply(lambda x: re.sub(r'[^\w\s,]', '', x))




