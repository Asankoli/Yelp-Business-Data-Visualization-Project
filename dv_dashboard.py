

import streamlit as st
import pandas as pd
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
import folium
from folium import plugins
from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud
# Page Configuration
st.set_page_config(page_title="Yelp Business Dashboard", page_icon=":bar_chart:", layout="wide", initial_sidebar_state='collapsed')

# import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from afinn import Afinn
import numpy as np
from collections import Counter


# Read Data
@st.cache_data
def get_data_from_drive(drive_link):
    output = 'data_business.csv'
    gdown.download(drive_link, output, quiet=True)
    df = pd.read_csv(output, header=0)
    return df



# Main Page
st.title(":bar_chart: Yelp Business Dashboard")
tab1,tab2,tab3 = st.tabs(['Overall','Rating','Sentimental Analysis'])

# Data Loading and Filtering
drive_link_business = "https://drive.google.com/uc?id=19eVp7E2a6agUN1Q1G2CdEdZ39_F-aR9T"
df = get_data_from_drive(drive_link_business)


# Sidebar
st.sidebar.header("Filter Options:")
# st.header(df.columns)
unique_cities = df['city'].unique()
city = st.sidebar.multiselect("Select City:", options=unique_cities, default=['Las Vegas'])
df_selection = df[df['city'].isin(city)]

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

def create_word_cloud(reviews):
    business_id = "4JNXUYY8wbaaDmk3BPzlWw"  # Replace with your business ID
    business_reviews = reviews[reviews['business_id'] == business_id]
    
    stop_words = set(stopwords.words('english'))
    word_list = ' '.join([word.lower() for text in business_reviews['text'] for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words and word.lower() != 'food' and word.lower() != 'restaurant'])
    
    word_counter = Counter(word_list.split())
    top_words = dict(word_counter.most_common(30))
    
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(top_words)

    # Display the word cloud in Streamlit
    st.image(wordcloud.to_array())


# Expander for Graphs
# with st.expander("Toggle Visibility"):
#     # Star Rating Distribution Graph
with tab2:
    col1,col2 = st.columns(2)
    with col1:
        st.header("Vegas Ratings Heatmap Animation")
        rating_data = df[['latitude', 'longitude', 'stars', 'review_count']]
        rating_data['popularity'] = rating_data['stars'] * rating_data['review_count']
        data = []
        stars_list = list(df_selection['stars'].unique())
        lat, lon, zoom_start = 36.127430, -115.138460, 11
        lon_min, lon_max = lon - 0.3, lon + 0.5
        lat_min, lat_max = lat - 0.4, lat + 0.5
        ratings_data_vegas = rating_data[
            (rating_data["longitude"] > lon_min) &
            (rating_data["longitude"] < lon_max) &
            (rating_data["latitude"] > lat_min) &
            (rating_data["latitude"] < lat_max)
        ]
        for star in stars_list:
            subset = ratings_data_vegas[ratings_data_vegas['stars'] == star]
            data.append(subset[['latitude', 'longitude']].values.tolist())
        m = folium.Map(location=[lat, lon], tiles="OpenStreetMap", zoom_start=zoom_start)
        hm = plugins.HeatMapWithTime(data, max_opacity=0.3, auto_play=True, display_index=True, radius=7)
        hm.add_to(m)

        folium_static(m, width=700, height=600)

    with col2:
        st.header("Star Rating Distribution")
        req = df_selection['stars'].value_counts().sort_index()
       
        st.bar_chart(req,width=0,height=600)

# Vegas Ratings Heatmap Animation
    





# Map of All Las Vegas Restaurants
with tab1: 

    col1, col2,col3 = st.columns(3)

    with col2:
        # Basic basemap of the world
        fig, ax = plt.subplots(figsize=(15, 6))

        # Use ortho projection for the globe type version
        m1 = Basemap(projection='ortho', lat_0=20, lon_0=-50, ax=ax)

        # Hex codes from google maps color palette = http://www.color-hex.com/color-palette/9261
        # Add continents
        m1.fillcontinents(color='#bbdaa4', lake_color='#4a80f5')
        # Add the oceans
        m1.drawmapboundary(fill_color='#4a80f5')
        # Draw the boundaries of the countries
        m1.drawcountries(linewidth=0.1, color="black")

        # Add the scatter points to indicate the locations of the businesses
        mxy = m1(df["longitude"].tolist(), df["latitude"].tolist())
        m1.scatter(mxy[0], mxy[1], s=3, c="orange", lw=3, alpha=1, zorder=5)

        plt.title("World-wide Yelp Reviews")
        st.pyplot(fig)

    
        



def get_data_from_drive_review(drive_link):
    output = 'data_review.csv'
    gdown.download(drive_link, output, quiet=True)
    df = pd.read_csv(output, header=0)
    return df
drive_link_review = "https://drive.google.com/uc?id=1QP1nBDNw6mW-uj5RynsLvnnYRuj4s_Lp"
df_review = get_data_from_drive_review(drive_link_review)


def generate_word_count_chart(reviews):
    business_id = "4JNXUYY8wbaaDmk3BPzlWw"  # Replace with your business ID
    business_reviews = reviews[reviews['business_id'] == business_id]
    
    stop_words = set(stopwords.words('english'))
    business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words and word.lower() not in ['food', 'restaurant']]))

    word_counts = business_reviews['text'].str.split(expand=True).stack().value_counts().reset_index()
    word_counts.columns = ['word', 'n']
    top_words = word_counts.head(10)


    st.subheader('Word Count')
    st.bar_chart(top_words.set_index('word'),height=450,width=0)


with tab3:
    col1, col2,col3 = st.columns(3)
    with col1:
        
        generate_word_count_chart(df_review)

    with col2:
        

       
            

        def positive_words_bar_graph(reviews, business_id):
            # Tokenize and remove stopwords
            business_reviews = reviews[reviews['business_id'] == business_id]
            stop_words = set(stopwords.words('english'))
            business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))
            
            afinn = Afinn()

            # Tokenize and count occurrences
            contributions = business_reviews['text'].apply(lambda x: word_tokenize(x)).explode().value_counts().reset_index()
            contributions.columns = ['word', 'occurrences']

            # Calculate contributions using Afinn scores
            contributions['score'] = contributions['word'].apply(lambda word: afinn.score(word))
            
            # Get top 20 positive and negative words
            top_words = contributions.groupby('word')['score'].sum().reset_index()
            
            top_words = pd.concat([top_words.nlargest(20, 'score'), top_words.nsmallest(20, 'score')], ignore_index=True)
            # top_words = top_words.nlargest(20, 'score').append(top_words.nsmallest(20, 'score'))
            # st.write(top_words)
            # Plot bar graph
            fig, ax = plt.subplots(figsize=(10,8))
            colors = np.where(top_words['score'] > 0, 'green', 'red')
            sns.barplot(x='score', y='word', data=top_words, palette=colors)
            ax.set_xlabel('Score')
            ax.set_ylabel('Word')
            # ax.set_title('Top 20 Positive and Negative Words Contribution')

            return fig



        # Example usage
        # Replace this with your actual reviews DataFrame and business_id
        reviews = df_review[['business_id', 'text']] 
        business_id = "4JNXUYY8wbaaDmk3BPzlWw"

        # Create the bar graph
        st.subheader("Score of Top 20 Words")
        fig = positive_words_bar_graph(reviews, business_id)
        st.pyplot(fig)
        # Show the plot in Streamlit

    with col3:
        st.subheader('Word Cloud for A specific business')
        create_word_cloud(df_review)
        




with tab3:
    col1, col2, col3 = st.columns(3)
    with col2:
        # Assuming 'reviews' is your DataFrame with 'business_id' and 'text' columns
        business_id = "4JNXUYY8wbaaDmk3BPzlWw"

        # Filter reviews for a specific business_id
        business_reviews = df_review[df_review['business_id'] == business_id]
        # st.title(len(business_reviews))
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words and word.lower() not in ['food', 'restaurant']]))

        # Count word occurrences
        word_counts = business_reviews['text'].str.split(expand=True).stack().value_counts().reset_index()
        word_counts.columns = ['word', 'n']

        # Sort and get top 10 words
        top_words = word_counts.head(10)
        
        # Plot the bar chart
        fig, ax = plt.subplots()
        ax.barh(top_words['word'], top_words['n'], color='skyblue')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Word')
        ax.set_title('Word Count')

        # Show the plot in Streamlit
        st.pyplot(fig)
    with col3:
        

        
            

        def positive_words_bar_graph(reviews, business_id):
            # Tokenize and remove stopwords
            business_reviews = reviews[reviews['business_id'] == business_id]
            stop_words = set(stopwords.words('english'))
            business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))
            
            afinn = Afinn()

            # Tokenize and count occurrences
            contributions = business_reviews['text'].apply(lambda x: word_tokenize(x)).explode().value_counts().reset_index()
            contributions.columns = ['word', 'occurrences']

            # Calculate contributions using Afinn scores
            contributions['score'] = contributions['word'].apply(lambda word: afinn.score(word))
            
            # Get top 20 positive and negative words
            top_words = contributions.groupby('word')['score'].sum().reset_index()
            
            top_words = pd.concat([top_words.nlargest(20, 'score'), top_words.nsmallest(20, 'score')], ignore_index=True)
            # top_words = top_words.nlargest(20, 'score').append(top_words.nsmallest(20, 'score'))
            # st.write(top_words)
            # Plot bar graph
            fig, ax = plt.subplots()
            colors = np.where(top_words['score'] > 0, 'green', 'red')
            sns.barplot(x='score', y='word', data=top_words, palette=colors)
            ax.set_xlabel('Score')
            ax.set_ylabel('Word')
            ax.set_title('Top 20 Positive and Negative Words Contribution')

            return fig



        # Example usage
        # Replace this with your actual reviews DataFrame and business_id
        reviews = df_review[['business_id', 'text']] 
        business_id = "4JNXUYY8wbaaDmk3BPzlWw"

        # Create the bar graph
        fig = positive_words_bar_graph(reviews, business_id)
        st.pyplot(fig)
        # Show the plot in Streamlit
        



# Hide Streamlit Style
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)  