import requests
import pandas as pd
from bs4 import BeautifulSoup
import scrapy 
#from pathlib import path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from langdetect import detect
from googletrans import Translator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def translate_to_english(text):
    try:
        # Detect the language of the text
        lang = detect(text)
        
        # If it's not in English, translate it to English
        if lang != 'en':
            translator = Translator()
            translated_text = translator.translate(text, src=lang, dest='en').text
            return translated_text
        else:
            return text
    except:
        return text 
    

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back into text
    return ' '.join(tokens)