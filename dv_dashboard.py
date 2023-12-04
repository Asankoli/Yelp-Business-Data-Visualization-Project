# import streamlit as st
# import pandas as pd
# import gdown
# import matplotlib.pyplot as plt
# import seaborn as sns
# from folium.plugins import HeatMap
# from streamlit_folium import folium_static
# import folium
# from folium import plugins

# # Streamlit App
# st.title("Data Visualization Dashboard")

# # Input field for Google Drive link
# # drive_link = st.text_input("Enter Google Drive link for Yelp Business data", key="drive_link")
# drive_link = "https://drive.google.com/uc?id=19eVp7E2a6agUN1Q1G2CdEdZ39_F-aR9T"

# # Check if a link is provided
# if drive_link:
#     try:
#         # Download the Yelp Business data from Google Drive
#         output = 'data.csv'
#         gdown.download(drive_link, output, quiet=True)

#         # Load data from the downloaded file
#         df = pd.read_csv(output, header=0)

#         # Print summary information about the dataset
#         st.write("### Yelp Business Data Summary")
#         st.write(df.info())

#         # # Trim dataset to only Las Vegas businesses
#         # st.write("### Trimmed Dataset - Las Vegas Businesses")
#         # st.write("Number of observations in the city category with Las Vegas:", len(df[df['city'] == "Las Vegas"]))

#         st.write("### Top 10 Business Categories")
#         categories = df['categories'].str.split(';')
#         # st.write(categories.head())
#         categories = df['categories'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).reset_index(name='Name')
#         # st.write(categories.head())
#         # Count occurrences of each category
#         category_counts = categories['Name'].value_counts().sort_values(ascending=False)
#         # st.write(category_counts.head())
#         # Select top 10 categories
#         top_categories = category_counts.head(10)

#         # Plotting
#         st.write("#### Bar Plot of Top 10 Business Categories")
#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax = sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
#         plt.figure(figsize=(10, 6))
#         plt.xlabel('Count')
#         plt.ylabel('Category')
#         plt.title('Top 10 Categories of Business')
#         st.pyplot(fig)



#         # # Subset to Las Vegas businesses
#         vegasbusiness = df[df['city'] == "Las Vegas"]
#         # st.write("Number of observations after subsetting:", len(vegasbusiness))

#         # # Display the data
#         # st.write(vegasbusiness.head())

#         # # Data Cleaning: Rename observations in the City variable that include the words Las Vegas to "Las Vegas"
#         df['city'][df['city'].str.contains("Las Vegas", case=False, na=False)] = "Las Vegas"
#         vegasbusiness = df[df['city'] == "Las Vegas"]

#         # st.write("### Updated Dataset - Las Vegas Businesses")
#         # st.write("Number of observations after cleaning:", len(vegasbusiness))

#         # # Display the cleaned data
#         # st.write(vegasbusiness.head())

#         # # Subset to only include restaurants or places that sell food
#         vegasfood = vegasbusiness[vegasbusiness['categories'].str.contains("Food|Restaurants", case=False, na=False)]

#         # st.write("### Vegas Food Dataset - Restaurants")
#         # st.write("Number of restaurants in Las Vegas:", len(vegasfood))

#         # Display the Vegas Food dataset
#         # st.write(vegasfood.head())
#         # Map of All Las Vegas Restaurants
#         st.write("### Map of All Las Vegas Restaurants")
#         m = folium.Map(location=[36.14, -115.2], zoom_start=11, tiles="OpenStreetMap")
#         for index, row in vegasfood.iterrows():
#             folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=1, fill_opacity=0.6,
#                                 color='purple').add_to(m)
#         folium_static(m, width=725, height=500)

#         # Star Rating Distribution Graph
#         st.write("### Star Rating Distribution")
#         req = df['stars'].value_counts().sort_index()
#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax = sns.barplot(x=req.index, y=req.values, alpha=0.8)
#         plt.title("Star Rating Distribution")
#         plt.ylabel('# of businesses', fontsize=12)
#         plt.xlabel('Star Ratings', fontsize=12)
#         st.pyplot(fig)

#         # Vegas Ratings Heatmap Animation
#         st.write("### Vegas Ratings Heatmap Animation")
#         data = []
#         stars_list = list(df['stars'].unique())
#         lat = 36.127430
#         lon = -115.138460
#         zoom_start = 11
#         lon_min, lon_max = lon - 0.3, lon + 0.5
#         lat_min, lat_max = lat - 0.4, lat + 0.5
#         ratings_data_vegas = df[
#             (df["longitude"] > lon_min) &
#             (df["longitude"] < lon_max) &
#             (df["latitude"] > lat_min) &
#             (df["latitude"] < lat_max)
#         ]
#         for star in stars_list:
#             subset = ratings_data_vegas[ratings_data_vegas['stars'] == star]
#             data.append(subset[['latitude', 'longitude']].values.tolist())

#         m = folium.Map(location=[lat, lon], tiles="OpenStreetMap", zoom_start=zoom_start)
#         hm = plugins.HeatMapWithTime(data, max_opacity=0.3, auto_play=True, display_index=True, radius=7)
#         hm.add_to(m)
#         folium_static(m, width=725, height=500)

#         # # Map of All Las Vegas Restaurants
#         # st.write("### Map of All Las Vegas Restaurants")
#         # m = folium.Map(location=[36.14, -115.2], zoom_start=11, tiles="OpenStreetMap")
#         # for index, row in vegasfood.iterrows():
#         #     folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=1, fill_opacity=0.6,
#         #                         color='purple').add_to(m)
#         # folium_static(m, width=800, height=600)

#     except Exception as e:
#         st.error(f"Error: {e}")



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
        # fig_stars = plt.figure(figsize=(8, 8))
        # ax = sns.barplot(x=req.index, y=req.values, alpha=0.8)
        # plt.title("Star Rating Distribution")
        # plt.ylabel('# of businesses', fontsize=12)
        # plt.xlabel('Star Ratings', fontsize=12)
        # rects = ax.patches
        # labels = req.values
        # for rect, label in zip(rects, labels):
        #     height = rect.get_height()
        #     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
        # st.pyplot(fig_stars)
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
    # with col1:
    #             # Sample it down to only the North America region 
    #     lon_min, lon_max = -132.714844, -59.589844
    #     lat_min, lat_max = 13.976715, 56.395664

    #     # Create the selector
    #     idx_NA = (df["longitude"] > lon_min) & (df["longitude"] < lon_max) & (df["latitude"] > lat_min) & (df["latitude"] < lat_max)

    #     # Apply the selector to subset
    #     NA_business = df[idx_NA]

    #     # Initiate the figure
    #     st.pyplot(plt.figure(figsize=(12, 6)))
    #     m2 = Basemap(
    #         projection='merc',
    #         llcrnrlat=lat_min,
    #         urcrnrlat=lat_max,
    #         llcrnrlon=lon_min,
    #         urcrnrlon=lon_max,
    #         lat_ts=35,
    #         resolution='i'
    #     )

    #     m2.fillcontinents(color='#191919', lake_color='#000000')  # dark grey land, black lakes
    #     m2.drawmapboundary(fill_color='#000000')  # black background
    #     m2.drawcountries(linewidth=0.1, color="w")  # thin white line for country borders

    #     # Plot the data
    #     mxy = m2(NA_business["longitude"].tolist(), NA_business["latitude"].tolist())
    #     m2.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.05, zorder=5)

    #     # Title for Streamlit app
    #     st.title("North America Region")
        


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
        # Assuming 'reviews' is your DataFrame with 'business_id' and 'text' columns
        # business_id = "4JNXUYY8wbaaDmk3BPzlWw"

        # # Filter reviews for a specific business_id
        # business_reviews = df_review[df_review['business_id'] == business_id]
        # # st.title(len(business_reviews))
        # # Tokenize and remove stopwords
        # stop_words = set(stopwords.words('english'))
        # business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words and word.lower() not in ['food', 'restaurant']]))

        # # Count word occurrences
        # word_counts = business_reviews['text'].str.split(expand=True).stack().value_counts().reset_index()
        # word_counts.columns = ['word', 'n']

        # # Sort and get top 10 words
        # top_words = word_counts.head(10)
        
        # # Plot the bar chart
        # fig, ax = plt.subplots()
        # ax.barh(top_words['word'], top_words['n'], color='skyblue')
        # ax.set_xlabel('Word Count')
        # ax.set_ylabel('Word')
        # ax.set_title('Word Count')

        # # Show the plot in Streamlit
        # st.pyplot(fig)
        generate_word_count_chart(df_review)

    with col2:
        

        # def positive_words_bar_graph(reviews, business_id):
        #     # Assuming 'reviews' is your DataFrame with 'business_id' and 'text' columns
        #     business_reviews = reviews[reviews['business_id'] == business_id]

        #     # Tokenize and remove stopwords
        #     stop_words = set(stopwords.words('english'))
        #     afinn = Afinn()

        #     business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

        #     # Calculate Afinn scores
        #     business_reviews['score'] = business_reviews['text'].apply(lambda x: afinn.score(x))

        #     # Count word occurrences and calculate contributions
        #     contributions = business_reviews.groupby('text')['score'].sum().reset_index()
        #     contributions.columns = ['word', 'contribution']

        #     # Get top 20 positive words
        #     top_positive_words = contributions.sort_values(by='contribution', ascending=False).head(20)
        #     st.write(top_positive_words)
        #     # Plot bar graph
        #     fig, ax = plt.subplots()
        #     colors = top_positive_words['contribution'].apply(lambda x: 'green' if x > 0 else 'red')
        #     ax.barh(top_positive_words['word'], top_positive_words['contribution'], color=colors)
        #     ax.set_xlabel('Contribution')
        #     ax.set_ylabel('Word')
        #     ax.set_title('Positive Words Contribution')

        #     return fig
        # def positive_words_bar_graph(reviews, business_id):
        #     # Assuming 'reviews' is your DataFrame with 'business_id' and 'text' columns
        #     business_reviews = reviews[reviews['business_id'] == business_id]
            
        #     # Tokenize and remove stopwords
        #     stop_words = set(stopwords.words('english'))
        #     business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))
            
        #     # Count word occurrences and calculate contributions
        #     contributions = business_reviews['text'].str.split(expand=True).stack().value_counts().reset_index()
        #     contributions.columns = ['word', 'occurrences']
            
        #     afinn = pd.DataFrame({'word': list(stopwords.words('english')), 'score': 0})
        #     contributions = pd.merge(contributions, afinn, how='left', on='word')
        #     contributions['contribution'] = contributions['occurrences'] * contributions['score']
            
        #     # Get top 20 positive words
        #     top_positive_words = contributions.sort_values(by='contribution', ascending=False).head(20)
        #     st.title(len(top_positive_words))
        #     st.write(top_positive_words)
            

        #     # Plot bar graph
        #     fig, ax = plt.subplots()
        #     colors = top_positive_words['contribution'].apply(lambda x: 'green' if x > 0 else 'red')
        #     ax.barh(top_positive_words['word'], top_positive_words['contribution'], color=colors)
        #     ax.set_xlabel('Contribution')
        #     ax.set_ylabel('Word')
        #     ax.set_title('Positive Words Contribution')

        #     return fig
        
        # def positive_words_bar_graph(reviews, business_id):
        #     # Tokenize and remove stopwords
        #     business_reviews = reviews[reviews['business_id'] == business_id]
        #     stop_words = set(stopwords.words('english'))
        #     business_reviews['text'] = business_reviews['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))
            
        #     afinn = Afinn()

        #     # Tokenize and count occurrences
        #     contributions = business_reviews['text'].apply(lambda x: word_tokenize(x)).explode().value_counts().reset_index()
        #     contributions.columns = ['word', 'occurrences']

        #     # Calculate contributions using Afinn scores
        #     contributions['score'] = contributions['word'].apply(lambda word: afinn.score(word))
            

        #     # Get top 20 positive words
        #     top_positive_words = contributions.groupby('word')['score'].sum().reset_index().nlargest(20, 'score')
        #     top_positive_words.reset_index()
        #     st.write(top_positive_words)
            
        #     # Plot bar graph
        #     fig, ax = plt.subplots()
        #     colors = np.where(top_positive_words['score'] > 0, 'green', 'red')
        #     sns.barplot(x='score', y='word', data=top_positive_words, palette=colors)
        #     ax.set_xlabel('Score')
        #     ax.set_ylabel('Word')
        #     ax.set_title('Positive Words Contribution')

        #     return fig
            

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
        


# Hide Streamlit Style
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)  