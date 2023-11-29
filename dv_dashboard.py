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



# Install necessary packages
# !pip install streamlit pandas gdown seaborn matplotlib folium streamlit_folium

import streamlit as st
import pandas as pd
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
import folium
from folium import plugins
from mpl_toolkits.basemap import Basemap
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer



import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Page Configuration
st.set_page_config(page_title="Yelp Business Dashboard", page_icon=":bar_chart:", layout="wide", initial_sidebar_state='collapsed')
st.set_option('deprecation.showPyplotGlobalUse',False)

# Read Data
@st.cache_data
def get_data_from_drive(drive_link):
    output = 'data.csv'
    gdown.download(drive_link, output, quiet=True)
    df = pd.read_csv(output, header=0)
    return df



# Main Page
st.title(":bar_chart: Yelp Business Dashboard")
tab1,tab2,tab3 = st.tabs(['Overall','Rating','Sentimental Analysis'])

@st.cache_data
def load_data(url):
    df = get_data_from_drive(url)
    return df

# @st.cache_data
# def load_data_local(url):
#     df =  pd.read_csv(url)
#     return df


# Data Loading and Filtering
business_link = "https://drive.google.com/uc?id=19eVp7E2a6agUN1Q1G2CdEdZ39_F-aR9T"
review_link = "https://drive.google.com/uc?id=1QP1nBDNw6mW-uj5RynsLvnnYRuj4s_Lp"
df = load_data(business_link)
review_df = load_data(review_link)


# Sidebar
st.sidebar.header("Filter Options:")
unique_cities = df['city'].unique()
city = st.sidebar.multiselect("Select City:", options=unique_cities, default=['Las Vegas'])
df_selection = df[df['city'].isin(city)]


def plot_useful_reviews(mon_ami_gabi_reviews):
    useful_reviews = mon_ami_gabi_reviews.groupby('useful').size().reset_index(name='Count')
    useful_reviews = useful_reviews.sort_values(by='Count', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(useful_reviews['useful'].astype(str), useful_reviews['Count'], color='skyblue')
    for rect, count in zip(bar_plot, useful_reviews['Count']):
        plt.text(rect.get_x() + rect.get_width() / 2, count + 0.5, f"({count})", ha='center', va='bottom')
    
    plt.xlabel('Useful Reviews')
    plt.ylabel('Count')
    plt.title('Useful Reviews and Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words and word not in ['food', 'restaurant']]
    return words

def plot_word_count(business_id):
    reviews_filtered = review_df[review_df['business_id'] == business_id]
    reviews_filtered['text'] = reviews_filtered['text'].apply(preprocess_text)
    words = [word for sublist in reviews_filtered['text'] for word in sublist]
    word_counts = Counter(words)
    most_common_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(most_common_words.keys(), most_common_words.values(), color='skyblue')
    for rect, count in zip(bar_plot, most_common_words.values()):
        plt.text(rect.get_x() + rect.get_width() / 2, count + 0.5, f"({count})", ha='center', va='bottom')

    plt.xlabel('Word')
    plt.ylabel('Word Count')
    plt.title('Word Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

def positive_words_bar_graph(reviews, business_id):
    sia = SentimentIntensityAnalyzer()
    reviews_filtered = reviews[reviews['business_id'] == business_id]
    reviews_filtered['text'] = reviews_filtered['text'].apply(lambda x: word_tokenize(x.lower()))
    positive_words = {}
    for text in reviews_filtered['text']:
        for word in text:
            if sia.polarity_scores(word)['compound'] > 0:
                positive_words[word] = positive_words.get(word, 0) + 1

    sorted_positive_words = dict(sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:20])
    
    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(sorted_positive_words.keys(), sorted_positive_words.values(), color='lightgreen')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Top Positive Words')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

# Expander for Graphs
# with st.expander("Toggle Visibility"):
#     # Star Rating Distribution Graph
with tab2:
    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Vegas Ratings Heatmap Animation")
        rating_data = df[['latitude', 'longitude', 'stars', 'review_count']]
        rating_data.loc['popularity'] = rating_data['stars'] * rating_data['review_count']
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
        folium_static(m, width=500, height=300)

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
        st.bar_chart(req)

# Vegas Ratings Heatmap Animation
    


# Map of All Las Vegas Restaurants
with tab1: 
    # st.header("Map of All Las Vegas Restaurants")
    # m_restaurants = folium.Map(location=[36.14, -115.2], zoom_start=11, tiles="OpenStreetMap")
    # for index, row in df_selection.iterrows():
    #     folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=1, fill_opacity=0.6,
    #                         color='purple').add_to(m_restaurants)
    # folium_static(m_restaurants, width=300, height=300)
    #basic basemap of the world

# Assuming 'business' is your DataFrame with latitude and longitude columns
    # Define the figure size


    # Use ortho projection for the globe type version
    #  Set the size of the map
    # Set the size of the map
    map_width = 800  # Specify the width in pixels
    map_height = 600  # Specify the height in pixels

    # Set up the Basemap
    fig, ax = plt.subplots(figsize=(map_width / 100, map_height / 100))
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

    # Set title
    st.title("World-wide Yelp Reviews")

    # Show the plot in Streamlit
    st.pyplot(fig)

    
with tab3:
    st.header('Sentimental Analysis of reviews')
    mon_ami_gabi_reviews = review_df[review_df.business_id== "4JNXUYY8wbaaDmk3BPzlWw"]
    st.write(mon_ami_gabi_reviews)
    plot_useful_reviews(mon_ami_gabi_reviews)
    plot_word_count("4JNXUYY8wbaaDmk3BPzlWw")
    positive_words_bar_graph(review_df, "4JNXUYY8wbaaDmk3BPzlWw")


# Hide Streamlit Style
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)
