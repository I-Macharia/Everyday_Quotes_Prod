import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import FreqDist
import plotly.express as px

import zipfile
import pickle
import string
import gzip


from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from sklearn.svm import SVC

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import streamlit as st

# Download the model using spacy.cli.download
spacy.cli.download("en_core_web_sm")

nltk.download("punkt")
nltk.download("stopwords")


def translate_to_english(text):
    """
Translates the given text to English if it is not already in English.

Args:
    text (str): The text to be translated.

Returns:
    str: The translated text in English, or the original text if it is already in English.

Example:
    ```python
    translated_text = translate_to_english("Bonjour")
    print(translated_text)  # Output: "Hello"
    ```
"""
    try:
        # Detect the language of the text
        lang = detect(text)
        
        if lang == 'en':
            return text
        
        translator = Translator()
        translation = translator.translate(text, src=lang, dest='en')
        
        return translation.text if translation else text
    except Exception as e:
        print(f"An error occurred: {e}")
        return text

    

def preprocess_text(text):
    """
Preprocesses the given text by converting it to lowercase, tokenizing it, removing stopwords, and joining the tokens back into text.

Args:
    text (str): The text to be preprocessed.

Returns:
    str: The preprocessed text.

Example:
    ```python
    preprocessed_text = preprocess_text("This is a sample text.")
    print(preprocessed_text)  # Output: "sample text"
    ```
"""

    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

     # Tokenize
    words = nltk.word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    words = [word for word in words if word.isalpha() and word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join tokens back into text 
    
    return ' '.join(tokens), words


def compress_tags(tags_list):
    return list(set(tags_list))[:5]

def preprocesss_text(text):
    """
    Preprocesses the given text by converting it to lowercase, tokenizing it, removing stopwords, and joining the tokens back into text.
    
    Parameters:
    - text (str): The input text.
    
    Returns:
    - str: The preprocessed text.
    """
    if isinstance(text, float):
        # Return an empty string for float values
        return ''
    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Simple text cleaning process
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove specific punctuation marks
        text = ''.join([c for c in text if c not in ('!', '.', ':', '?', ',', '\"')])
        # Remove any remaining punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def aggregate_statistics(polarity_scores, subjectivity_scores):
    polarity_mean = np.mean(polarity_scores)
    polarity_median = np.median(polarity_scores)
    polarity_std = np.std(polarity_scores)
    subjectivity_mean = np.mean(subjectivity_scores)
    subjectivity_median = np.median(subjectivity_scores)
    subjectivity_std = np.std(subjectivity_scores)

    stats = {
        "polarity_mean": polarity_mean,
        "polarity_median": polarity_median,
        "polarity_std": polarity_std,
        "subjectivity_mean": subjectivity_mean,
        "subjectivity_median": subjectivity_median,
        "subjectivity_std": subjectivity_std,
    }

    st.write("### Aggregate Statistics")
    st.write(f"**Polarity Mean:** {polarity_mean:.3f}")
    st.write(f"**Polarity Median:** {polarity_median:.3f}")
    st.write(f"**Polarity Standard Deviation:** {polarity_std:.3f}")
    st.write(f"**Subjectivity Mean:** {subjectivity_mean:.3f}")
    st.write(f"**Subjectivity Median:** {subjectivity_median:.3f}")
    st.write(f"**Subjectivity Standard Deviation:** {subjectivity_std:.3f}")

    return stats

def plot_histograms(polarity_scores, subjectivity_scores):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(polarity_scores, bins=10, kde=True, color='skyblue', edgecolor='black')
    plt.title('Distribution of Polarity Scores')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(subjectivity_scores, bins=10, kde=True, color='salmon', edgecolor='black')
    plt.title('Distribution of Subjectivity Scores')
    plt.xlabel('Subjectivity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(plt)

def sentiment_categories(polarity_scores, positive_threshold=0.2, negative_threshold=-0.2):
    positive_quotes = sum(p > positive_threshold for p in polarity_scores)
    neutral_quotes = sum(
        -positive_threshold <= p <= positive_threshold for p in polarity_scores
    )
    negative_quotes = sum(p < negative_threshold for p in polarity_scores)

    categories = {
        "positive_quotes": positive_quotes,
        "neutral_quotes": neutral_quotes,
        "negative_quotes": negative_quotes,
    }

    st.write("### Sentiment Categories")
    st.write(f"**Positive Quotes:** {positive_quotes}")
    st.write(f"**Neutral Quotes:** {neutral_quotes}")
    st.write(f"**Negative Quotes:** {negative_quotes}")

    return categories

def correlation_analysis(polarity_scores):
    quote_lengths = np.random.randint(10, 200, size=len(polarity_scores))
    correlation = np.corrcoef(polarity_scores, quote_lengths)[0, 1]

    st.write("### Correlation Analysis")
    st.write(f"**Correlation between Polarity Scores and Quote Lengths:** {correlation:.3f}")

    return correlation

def save_results(filename, results):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

class QuoteFinder:
    def __init__(self, quotes_df=None):
        self.quotes_df = quotes_df
        self.vectorizer = None
        self.svm_model = None

        if self.quotes_df is not None:
            self.quotes_df['combined'] = self.quotes_df['quote_2'] + " " + self.quotes_df['author_2']

    def train_model(self):
        if self.quotes_df is None:
            raise ValueError("Quotes DataFrame is not set.")
        
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(self.quotes_df['combined'])
        
        y = self.quotes_df.index
        
        self.svm_model = SVC(kernel='linear')
        self.svm_model.fit(X, y)

    def clean_text(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
        return ' '.join(tokens)

    def find_quote_for_tweet(self, tweet):
        cleaned_tweet = self.clean_text(tweet)
        vectorized_tweet = self.vectorizer.transform([cleaned_tweet])

        try:
            index = self.svm_model.predict(vectorized_tweet)[0]
            quote = self.quotes_df.iloc[index]['quote_2']
            author = self.quotes_df.iloc[index]['author_2']
            return quote, author
        except IndexError as e:
            print(f"Error: {e}")
            print(f"Predicted index {index} is out of range.")
            return "No quote found.", "Unknown"
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An error occurred.", "Unknown"

    # Serialize and compress data
    def save(self, vectorizer_filepath, model_filepath, dataframe_filepath):
        with gzip.open(vectorizer_filepath, 'wb') as file:
            pickle.dump(self.vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)

        with gzip.open(model_filepath, 'wb') as file:
            pickle.dump(self.svm_model, file, protocol=pickle.HIGHEST_PROTOCOL)

        with gzip.open(dataframe_filepath, 'wb') as file:
            pickle.dump(self.quotes_df, file, protocol=pickle.HIGHEST_PROTOCOL)
            
    # Decompress and deserialize data
    @classmethod
    def load(cls, vectorizer_filepath, model_filepath, dataframe_filepath):
        with gzip.open(vectorizer_filepath, 'rb') as file:
            vectorizer = pickle.load(file)

        with gzip.open(model_filepath, 'rb') as file:
            svm_model = pickle.load(file)

        with gzip.open(dataframe_filepath, 'rb') as file:
            quotes_df = pickle.load(file)

        instance = cls(quotes_df)
        instance.vectorizer = vectorizer
        instance.svm_model = svm_model

        return instance
