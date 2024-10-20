import streamlit as st
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import pickle
import pandas as pd

# Load the sentiment prediction model
with open("./model/model_predi.pkl", "rb") as model_file:
    model = pickle.load(model_file)

df_stopwords = pd.read_csv('./data/stopword_filter.csv', header=None, names=['stopword'])
list_stopwords = df_stopwords['stopword'].tolist()

df = pd.read_csv('./data/comments_predicted_sentiment.csv')

# Sidebar navigation
menu = st.sidebar.selectbox('Select an Option', ['Home','Sentiment Analysis', 'Brief Analysis'])

# Function to generate word cloud
def generate_wordcloud(comments, stopwords):
    wordcloud = WordCloud(
        width=600, height=300,  # Adjusted size
        background_color='white',
        stopwords=stopwords,
        colormap='viridis',
        random_state=42
    ).generate(comments)
    return wordcloud

# Function to get top N words
def get_top_n_words(dataframe, stopwords, n=10):
    all_comments = ' '.join(dataframe['cleaned_comment'])
    words = [word for word in all_comments.split() if word not in stopwords]
    word_count = Counter(words)
    return word_count.most_common(n)

# Main app
if menu == 'Home':
    st.title("Tentang Aplikasi Streamlit Ini")
    st.write("""
        Aplikasi web ini dirancang untuk melakukan analisis sentimen pada data teks. 
        Aplikasi ini menggunakan model yang dilatih dengan dataset komentar untuk 
        mengklasifikasikan sentimen menjadi tiga kategori: Positif, Netral, dan Negatif. 

        Anda dapat memasukkan teks Anda sendiri untuk dianalisis, melihat word cloud, 
        dan menganalisis distribusi sentimen dari komentar dalam dataset.
    """)

    st.write("### Pratayang Video yang digunakan")
    st.video("https://www.youtube.com/watch?v=PGtfI4ugjnE&ab_channel=GhibranArrazi")

elif menu == 'Sentiment Analysis':
    st.title("Sentiment Analysis NLP")
    st.subheader("Streamlit Projects")

    # User input form
    with st.form("nlpForm"):
        raw_text = st.text_area("Masukkan Teks di sini")
        submit_button = st.form_submit_button(label='Analyze')

        if submit_button:
            if raw_text:
                # Display overall sentiment prediction
                st.info("Results")
                
                # Predict sentiment for the entire input
                overall_sentiment = model.predict([raw_text])[0]
                
                # Directly use the prediction label ('positive', 'neutral', 'negative')
                st.write(f"Overall Sentiment: {overall_sentiment}")
            else:
                st.warning("Please enter some text.")

elif menu == 'Brief Analysis':
    st.title("Word Cloud and Top Words per Sentiment")

    # Create boxes at the top of the screen
    total_comments = len(df)
    most_common_word = get_top_n_words(df, list_stopwords, 1)[0][0] if total_comments > 0 else "N/A"
    positive_count = df[df['predicted_sentiment'] == 'positive'].shape[0]
    negative_count = df[df['predicted_sentiment'] == 'negative'].shape[0]
    neutral_count = df[df['predicted_sentiment'] == 'neutral'].shape[0]

    # Create columns for the stats
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Comments", total_comments)
    with col2:
        st.metric("Most Common Word", most_common_word)
    with col3:
        st.metric("Total Positive", positive_count)
    with col4:
        st.metric("Total Negative", negative_count)
    with col5:
        st.metric("Total Neutral", neutral_count)

    # Sentiment selection box
    sentiment_option = st.selectbox("Select Sentiment", ['All', 'Positive', 'Neutral', 'Negative'])

    # Filter comments based on selected sentiment
    if sentiment_option == 'Positive':
        selected_comments = df[df['predicted_sentiment'] == 'positive']
    elif sentiment_option == 'Neutral':
        selected_comments = df[df['predicted_sentiment'] == 'neutral']
    elif sentiment_option == 'Negative':
        selected_comments = df[df['predicted_sentiment'] == 'negative']
    else:
        selected_comments = df  # For 'All', use the entire dataset

    all_comments = ' '.join(selected_comments['cleaned_comment'])

    # Generate word cloud
    wordcloud = generate_wordcloud(all_comments, list_stopwords)
    
    # Display the word cloud
    st.write("Word Cloud for Selected Sentiment:")
    plt.figure(figsize=(10, 5))  # Adjusted size for better visibility
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Get top N words for the selected sentiment
    st.write(f"Top Words for {sentiment_option} Comments:")
    top_n_words = get_top_n_words(selected_comments, list_stopwords)
    
    # Create barplot for top N words
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjusted size
    sns.barplot(x=[x[1] for x in top_n_words], y=[x[0] for x in top_n_words], ax=ax)
    ax.set_title(f'Top Words in {sentiment_option} Comments')
    ax.set_xlabel('Count')
    ax.set_ylabel('Words')
    st.pyplot(fig)

    # Display the sentiment distribution as a pie chart when "All" is selected
    if sentiment_option == 'All':
        st.write("Sentiment Distribution:")

        # Sentiment distribution pie chart
        sentiment_counts = df['predicted_sentiment'].value_counts()
        
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjusted size
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['yellow', 'red', 'green'], startangle=90)
        ax.set_title('Sentiment Distribution')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Display the pie chart in Streamlit
        st.pyplot(fig)

elif menu == "Menu 3":
    st.write("Menu 3 content goes here.")
