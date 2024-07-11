import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import streamlit as st
import pickle
import joblib
from PIL import Image, ImageDraw, ImageFont
from my_functions import (load_results, QuoteFinder,aggregate_statistics,
    plot_histograms,
    sentiment_categories,
    correlation_analysis)
from chatbot import ChatBot
import spacy
import spacy.cli

# Download the model using spacy.cli.download
spacy.cli.download("en_core_web_sm")

#nltk.download("punkt")
#nltk.download("stopwords")


# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Function to load data
@st.cache_data
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




def main_page():
    st.title("Bobby's Quotes Bot")
      
    # Display instructions
    st.header("How to Use the ChatBot")
    st.write("""
    1. Type a greeting like "hello".
    2. Ask for the current time by typing "time".
    3. Ask for today's date by typing "date".
    4. Ask for a quote on a specific topic, for example, "quote about time".
    5. You can also say "thanks" to receive a polite response.
    6. Explore other predefined responses like "yes", "no", "what can you do", and "capabilities".
    7. You can go ahead and tell it whatever, and it will return a relatable quote 80% of the time.
    """)
    quotes_vectorized, tweets_vectorized, df, quotes_2, combined_text, tweets_df = load_data()

    
    data_handler = QuoteFinder.load('data/vectorizer.pkl.gz', 'data/svm_model.pkl.gz', 'data/quotes_df.pkl.gz')
    ai = ChatBot(name="Bobby", quote_finder=data_handler)

    user_input = st.text_input("You -->")
    if st.button("Send") and user_input:
        reply = ai.generate_response(user_input)
        
        st.write("Bobby -->")
        st.write(reply)


def about_page():
    st.title("Daily Motivation Quotes")
    st.write("""
In a world filled with daily challenges and responsibilities, staying motivated is essential for personal growth and well-being. This data science project is dedicated to curating and delivering a diverse collection of carefully selected quotes. These inspirational snippets, sourced from various outlets including historical figures, popular literature, and notable personalities, aim to provide a source of encouragement, reflection, and empowerment for individuals facing the hustle of everyday life.
#### Objectives:
1. Curate Inspirational Quotes: Gather a diverse collection of quotes from the Good Reads website, which boasts an extensive compilation of quotes spanning various genres and themes.
2. Daily Motivational Updates: Develop a system to provide users with daily updates featuring a thoughtfully chosen quote. These updates will cater to different areas of life, ensuring a comprehensive and relatable experience.
3. Tag-based Grouping: Implement a categorization mechanism that tags each quote based on its thematic content. This grouping will enable users to easily identify quotes that resonate with their specific preferences or current situations.
## Data Understanding
• Source quotes from the Good Reads website, exploring the wide array of authors and themes available.
• Analyze the structure of the collected data, including metadata such as author names, publication dates, and associated tags.
""")
    
    with open('data/quotes_2.pkl', 'rb') as qt:
        df = pickle.load(qt)
    
    def generate_word_cloud(data, column, title):
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(data[column]))
        plt.figure(figsize=(10, 5))
        st.header(title)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
    
    def author_contribution_bar_chart(data):
        author_counts = data['author_2'].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=author_counts.values, y=author_counts.index, legend=False, dodge=False)
        plt.title('Top 10 Authors Contribution')
        plt.xlabel('Number of Quotes')
        plt.ylabel('Author')
        st.pyplot(plt)
    
    tag_counts = df.groupby(by='Tags').size().nlargest(10).reset_index(name='Count')
    fig = px.pie(tag_counts, names='Tags', values='Count', title='Tags Distribution', custom_data=['Tags'])
    fig.update_traces(hovertemplate='Tag: %{customdata[0]}<br>Count: %{value}')
    
    st.title('Word Clouds for Quotes Data')
    generate_word_cloud(df, 'author', 'Word Cloud of Author Names')
    
    generate_word_cloud(df, 'quote_2', 'Word Cloud of Words in Quotes')
    st.title('Author Contribution Analysis')
    author_contribution_bar_chart(df)
    st.title('Tag Distribution Pie Chart')
    st.plotly_chart(fig)
    
    # Calculate and display aggregate statistics
    polarity_scores = df['polarity'].values  # Assuming polarity scores are in the 'polarity' column
    subjectivity_scores = df['subjectivity'].values  # Assuming subjectivity scores are in the 'subjectivity' column

    stats = aggregate_statistics(polarity_scores, subjectivity_scores)
    st.write("""
             Polarity indicates the sentiment orientation of the quote (positive, negative, or neutral), with values closer to 1 indicating more positive sentiment, closer to -1 indicating more negative sentiment, and around 0 indicating neutral sentiment. 
Subjectivity measures how subjective or opinionated the quote is, with values closer to 1 indicating more subjective content and around 0 indicating more objective or factual content.
""")
    
    # Display histograms
    plot_histograms(polarity_scores, subjectivity_scores)
    st.write("""
              Polarity Statistics:
    
    The mean polarity score is slightly positive (0.113), indicating that, on average, the quotes tend to be more positive than negative.
    The median polarity score is 0.050, suggesting that there is a right-skewed distribution with more quotes leaning towards positivity.
    The standard deviation of polarity scores is relatively high (0.290), indicating a wide variability in sentiment polarity among the quotes.
    
    Subjectivity Statistics:
    
    The mean subjectivity score is 0.450, indicating that the quotes, on average, contain a moderate level of subjectivity.
    The median subjectivity score is 0.500, suggesting an even distribution of subjective and objective quotes.
    The standard deviation of subjectivity scores is 0.302, indicating variability in the subjective nature of the quotes.
    
             """)
    
    # Display sentiment categories
    sentiment_categories(polarity_scores)
    st.write("This distribution shows that the majority of quotes tend to be neutral or slightly positive in sentiment.")
    
    # Display correlation analysis
    correlation_analysis(polarity_scores)
    st.write("This suggests that longer quotes tend to have slightly lower polarity scores, indicating that longer quotes may express less extreme sentiment.")
    
    st.header("Conclusion")
    st.write("""
    - The Project has been a success, having achieved our objectives and finding the following results:
        1. Curate Inspirational Quotes: we managed to scrape 3000 quotes from the website from several different industry experts and book authors.
        2. Build a Web App: we have built a web app that allows users to search for quotes and trained a model to recommend quotes based on user input.
        3. Tag-based Grouping: we assigned each quote a polarity score and, in effect, created tags for positive, neutral, and negative.
    - The project also raised several action areas. Specifically, when doing the modeling, we realized we needed more quotes to be able to reach the desired accuracy of 1 for the recommendations to be accurate. This also showed that to try better feature engineering, we would need to use either online code editors with greater compute power or alternatively upgrade our physical PC.
    - Optimization:
        1. Continued refining algorithms to enhance the relevance and diversity of quotes.
        2. Optimized the tagging system for improved categorization accuracy and user experience.
    """)

    st.header("Recommendations")
    st.write("""
    - These are some of the recommendations for moving forward with the next steps:
        1. Find a new source of quotes to increase the total number to at least 10,000 quotes.
        2. Create user login functionality and storage of favorite quotes to personalize the app.
        3. Update the Bot to allow sharing of quotes to other platforms.
    - Feedback Integration:
        1. Actively solicit user feedback through surveys, polls, and direct interactions.
        2. Integrate feedback into the development process to prioritize features and enhance the overall user experience.
    - Scalability Planning:
        1. Proactively plan for scalability by designing systems to accommodate larger volumes of users and data.
        2. Optimize database performance and leverage cloud infrastructure to handle increased traffic and demand.
    - Community Building:
        1. Foster a vibrant and engaged community through social media engagement and collaborative initiatives.
        2. Empower users to share experiences and insights, cultivating a supportive and inspiring community.
    - Continuous Improvement:
        1. Commit to iterating on the platform based on user feedback, emerging trends, and technological advancements.
        2. Stay agile and adaptive to ensure the platform remains relevant, impactful, and inspiring for users.
    """)
        
pages = {'About': about_page, 'Main': main_page}
page = st.sidebar.selectbox('Go to', list(pages.keys()))
pages[page]()
