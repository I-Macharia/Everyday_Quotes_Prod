import numpy as np
import pandas as pd
import datetime
import pickle
import spacy
import gzip
import joblib
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from my_functions import load_results, QuoteFinder
import spacy.cli 
import nltk

# nltk.download("punkt")
# nltk.download("stopwords")
# # Download the model using spacy.cli.download
# spacy.cli.download("en_core_web_sm")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to load data
def load_data():
    with open('data/quotes_vectorized.pickle', 'rb') as f:
        quotes_vectorized = pickle.load(f)
    with open('data/tweets_vectorized.pickle', 'rb') as q:
        tweets_vectorized = pickle.load(q)
    with open('data/quotes_2_pickle', 'rb') as d:
        quotes_2 = pickle.load(d)
    with open('data/tweets_df', 'rb') as t:
        tweets_df = pickle.load(t)
    df = pd.read_csv('data/quotes_2.csv')
    with open('data/combined_text', 'rb') as ct:
        combined_text = pickle.load(ct)
    return quotes_vectorized, tweets_vectorized, df, quotes_2, combined_text, tweets_df

quotes_vectorized, tweets_vectorized, df, quotes_2, combined_text, tweets_df = load_data()

# Load the QuoteFinder class with the saved components
quote_finder = QuoteFinder.load('data/vectorizer.pkl.gz', 'data/svm_model.pkl.gz', 'data/quotes_df.pkl.gz')


class ChatBot:
    def __init__(self, name, quote_finder):
        self.name = name
        self.quote_finder = quote_finder
        self.conversation_history = []
        self.responses = {
            "hello": self.get_hello_response,
            "thanks": self.get_thanks_response,
            "yes": self.get_yes_response,
            "no": self.get_no_response,
            "time": self.get_time_response,
            "date": self.get_date_response,
            "dev": self.get_dev_response,
            "hello dev": self.get_hello_dev_response,
            "what can you do": self.get_capabilities_response,
            "capabilities": self.get_capabilities_response,
            "quote about time": self.get_quote_about_time_response,
        }

    @staticmethod
    def wake_up(text):
        wake_words = ["hey", "hi", "hello", "dev", "assistant", "time", "date", "quote", "thanks", "yes", "goodbye", "what can you do", "capabilities"]
        synonyms = ["howdy", "greetings", "salutations", "morning", "afternoon", "evening"]
        all_wake_words = wake_words + synonyms
        return any(word in text.lower() for word in all_wake_words)

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

    @staticmethod
    def action_date():
        return datetime.datetime.now().date().strftime('%B %d, %Y')

    def get_hello_response(self):
        return f"Hello, my name is {self.name}!"

    def get_thanks_response(self):
        return "You're welcome! Is there anything else you would like me to help you with?"

    def get_yes_response(self):
        return "Great! How can I assist you today?"

    def get_no_response(self):
        return "Bye! Have a great day!"

    def get_time_response(self):
        return f"The time is {self.action_time()}!"

    def get_date_response(self):
        return f"Today's date is {self.action_date()}!"

    def get_dev_response(self):
        return "I'm happy to help you with whatever you need."

    def get_hello_dev_response(self):
        return "I'm doing well, how about you?"

    def get_capabilities_response(self):
        return (
            "I can assist you with various tasks such as telling the current time, providing today's date, "
            "offering quotes on specific topics, and more. Just let me know how I can help!"
        )

    def get_quote_about_time_response(self):
        quote, author = self.quote_finder.find_quote_for_tweet("time")
        return f"Quote: {quote}\nAuthor: {author}"
    
    def generate_response(self, text):
        doc = nlp(text)
        
        if "time" in [token.text for token in doc]:
            return self.get_time_response()
        elif any(command in text.lower() for command in self.responses):
            for command in self.responses:
                if command in text.lower():
                    return self.responses[command]()
        
        print("No specific command detected. Trying to find a quote.")
        # Default case: generate a quote for any input
        quote, author = self.quote_finder.find_quote_for_tweet(text)
        print(f"Quote: {quote}\n Author: {author}")
        return f"Quote: {quote} \n Author: {author}"

if __name__ == "__main__":
    quotes_vectorized, tweets_vectorized, df, quotes_2, combined_text, tweets_df = load_data()
    data_handler = QuoteFinder.load('data/vectorizer.pkl.gz', 'data/svm_model.pkl.gz', 'data/quotes_df.pkl.gz')
    ai = ChatBot(name="Bobby", quote_finder=data_handler)

    # Testing the chatbot with various inputs
    test_input = "Give me a quote about time."
    response = ai.generate_response(test_input)
    print(f"AI --> {response}")
