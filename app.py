# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from collections import Counter
# import itertools
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import re
# import emoji
# from langdetect import detect
# import spacy
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from googleapiclient.discovery import build
# import time
#
# # Download VADER lexicon
# nltk.download('vader_lexicon')
# sid = SentimentIntensityAnalyzer()
#
# # Load spacy model
# nlp = spacy.load("en_core_web_sm")
#
# st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")
# st.title("🎯 YouTube Video Sentiment Analysis with Word Clouds")
#
# # YouTube API key placeholder - replace with your API key
# API_KEY = "AIzaSyDwljpW9CNQI2E-0TomP-oZseC0j2gRsfk"
#
# # Default YouTube video ID
# DEFAULT_VIDEO_ID = "6QYcd7RggNU"
#
# def extract_video_id(url_or_id):
#     # Regex pattern to extract video ID from URL or return ID if string is already an ID
#     pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
#     match = re.search(pattern, url_or_id)
#     if match:
#         return match.group(1)
#     elif len(url_or_id) == 11 and re.match(r'^[0-9A-Za-z_-]{11}$', url_or_id):
#         return url_or_id
#     else:
#         return None
#
# def get_comments(youtube, video_id, max_results=500):
#     comments = []
#     next_page_token = None
#     while True:
#         request = youtube.commentThreads().list(
#             part="snippet",
#             videoId=video_id,
#             maxResults=100,
#             pageToken=next_page_token,
#             textFormat="plainText"
#         )
#         response = request.execute()
#
#         for item in response.get("items", []):
#             snippet = item["snippet"]["topLevelComment"]["snippet"]
#             comment = snippet.get("textDisplay")
#             author = snippet.get("authorDisplayName")
#             published = snippet.get("publishedAt")
#             like_count = snippet.get("likeCount")
#             comments.append({
#                 "video_id": video_id,
#                 "author": author,
#                 "published_at": published,
#                 "like_count": like_count,
#                 "comment_text": comment
#             })
#
#         next_page_token = response.get("nextPageToken")
#         if not next_page_token or len(comments) >= max_results:
#             break
#
#         time.sleep(1)  # avoid quota limits
#
#     return comments
#
# def clean_text(text):
#     text = str(text)
#     text = emoji.replace_emoji(text, '')  # Remove emojis
#     text = re.sub(r"http\S+", "", text)  # Remove URLs
#     text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
#     text = text.lower().strip()  # Lowercase and trim
#     return text
#
# def detect_lang(text):
#     try:
#         return detect(text)
#     except:
#         return 'unknown'
#
# def lemmatize_text(text):
#     doc = nlp(text)
#     return " ".join([token.lemma_ for token in doc if not token.is_stop])
#
# def label_sentiment(score):
#     if score > 0.05:
#         return 'Positive'
#     elif score < -0.05:
#         return 'Negative'
#     else:
#         return 'Neutral'
#
# def generate_wordcloud(text, colormap):
#     if not text.strip():
#         return None
#     wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
#     return wc
#
# def main():
#     st.sidebar.header("Input YouTube Video ID or URL")
#     user_input = st.sidebar.text_input("Enter YouTube video URL or ID:", value=f"https://www.youtube.com/watch?v={DEFAULT_VIDEO_ID}")
#
#     video_id = extract_video_id(user_input)
#     if video_id is None:
#         st.error("Invalid YouTube video URL or ID. Please enter a valid URL or 11-character video ID.")
#         return
#
#     # Set up YouTube API client
#     youtube = build("youtube", "v3", developerKey=API_KEY)
#
#     with st.spinner("Fetching comments and analyzing..."):
#         try:
#             comments_data = get_comments(youtube, video_id, max_results=500)
#         except Exception as e:
#             st.error(f"Error fetching comments: {e}")
#             return
#
#         if not comments_data:
#             st.error("No comments fetched for this video. Try a different one.")
#             return
#
#         df = pd.DataFrame(comments_data)
#         df['clean_text'] = df['comment_text'].apply(clean_text)
#         df['language'] = df['clean_text'].apply(detect_lang)
#         df = df[df['language'] == 'en'].copy()
#         df['lemmatized'] = df['clean_text'].apply(lemmatize_text)
#         df['sentiment_score'] = df['lemmatized'].apply(lambda x: sid.polarity_scores(x)['compound'])
#         df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)
#
#     st.markdown(f"### Analyzing Video ID: `{video_id}`")
#     st.metric("Total Comments Analyzed", len(df))
#
#     # Sentiment distribution
#     st.subheader("📈 Sentiment Distribution")
#     fig = px.histogram(df, x='sentiment_label', color='sentiment_label',
#                        category_orders={"sentiment_label": ["Positive", "Neutral", "Negative"]},
#                        title="Sentiment Distribution")
#     fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Count")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Word clouds per sentiment
#     st.subheader("☁️ Word Clouds by Sentiment")
#     col1, col2, col3 = st.columns(3)
#
#     # Separate texts by sentiment
#     positive_text = " ".join(df[df['sentiment_label']=="Positive"]['lemmatized'].tolist())
#     neutral_text = " ".join(df[df['sentiment_label']=="Neutral"]['lemmatized'].tolist())
#     negative_text = " ".join(df[df['sentiment_label']=="Negative"]['lemmatized'].tolist())
#
#     wc_pos = generate_wordcloud(positive_text, 'Greens')
#     wc_neu = generate_wordcloud(neutral_text, 'Greys')
#     wc_neg = generate_wordcloud(negative_text, 'Reds')
#
#     with col1:
#         st.markdown("**Positive Sentiment (Green Shades)**")
#         if wc_pos:
#             fig, ax = plt.subplots(figsize=(8, 4))
#             ax.imshow(wc_pos, interpolation='bilinear')
#             ax.axis('off')
#             st.pyplot(fig)
#         else:
#             st.write("No positive words available")
#
#     with col2:
#         st.markdown("**Neutral Sentiment (Grey Shades)**")
#         if wc_neu:
#             fig, ax = plt.subplots(figsize=(8, 4))
#             ax.imshow(wc_neu, interpolation='bilinear')
#             ax.axis('off')
#             st.pyplot(fig)
#         else:
#             st.write("No neutral words available")
#
#     with col3:
#         st.markdown("**Negative Sentiment (Red Shades)**")
#         if wc_neg:
#             fig, ax = plt.subplots(figsize=(8, 4))
#             ax.imshow(wc_neg, interpolation='bilinear')
#             ax.axis('off')
#             st.pyplot(fig)
#         else:
#             st.write("No negative words available")
#
#     # Top entities display - simplistic approach using noun chunks
#     st.subheader("🏷️ Top Entities Mentioned (Noun Phrases)")
#
#     def extract_entities(text):
#         doc = nlp(text)
#         return [chunk.text for chunk in doc.noun_chunks]
#     df['entities'] = df['lemmatized'].apply(extract_entities)
#
#     entity_list = list(itertools.chain.from_iterable(df['entities'].dropna().tolist()))
#     counter = Counter(entity_list)
#     top_n = 10
#     top_entities = counter.most_common(top_n)
#
#     if top_entities:
#         entities_df = pd.DataFrame(top_entities, columns=['Entity', 'Count'])
#         st.table(entities_df)
#
#         st.subheader("📊 Bar Chart for Top Entities")
#         fig, ax = plt.subplots(figsize=(12, 6))
#         sns.barplot(x='Count', y='Entity', data=entities_df, palette='viridis', ax=ax)
#         ax.set_title('Top Entities Mentioned')
#         ax.set_xlabel('Count')
#         ax.set_ylabel('Entity')
#         st.pyplot(fig)
#     else:
#         st.info("No entities found.")
#
#     # Show sample comments with sentiment
#     st.subheader("💬 Sample Comments")
#     display_cols = ['comment_text', 'sentiment_label', 'entities']
#     st.dataframe(df[display_cols].head(20).reset_index(drop=True))
#
#     # Download analyzed data as CSV
#     st.subheader("📥 Download Analyzed Comments (CSV)")
#     csv = df.to_csv(index=False).encode('utf-8')
#     st.download_button("Download CSV", csv, f"youtube_comments_{video_id}.csv", "text/csv")
#
# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import itertools
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import emoji
from langdetect import detect
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
import time

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Load spacy model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 YouTube Video Sentiment Analysis with Word Clouds")

# YouTube API key placeholder - replace with your API key
API_KEY = "AIzaSyDwljpW9CNQI2E-0TomP-oZseC0j2gRsfk"

# Default YouTube video ID
DEFAULT_VIDEO_ID = "6QYcd7RggNU"

def extract_video_id(url_or_id):
    # Regex pattern to extract video ID from URL or return ID if string is already an ID
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url_or_id)
    if match:
        return match.group(1)
    elif len(url_or_id) == 11 and re.match(r'^[0-9A-Za-z_-]{11}$', url_or_id):
        return url_or_id
    else:
        return None

def get_comments(youtube, video_id, max_results=500):
    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment = snippet.get("textDisplay")
            author = snippet.get("authorDisplayName")
            published = snippet.get("publishedAt")
            like_count = snippet.get("likeCount")
            comments.append({
                "video_id": video_id,
                "author": author,
                "published_at": published,
                "like_count": like_count,
                "comment_text": comment
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(comments) >= max_results:
            break

        time.sleep(1)  # avoid quota limits

    return comments

def clean_text(text):
    text = str(text)
    text = emoji.replace_emoji(text, '')  # Remove emojis
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    text = text.lower().strip()  # Lowercase and trim
    return text

def detect_lang(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def label_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def generate_wordcloud(text, colormap):
    if not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    return wc

def main():
    st.sidebar.header("Input YouTube Video ID or URL")
    user_input = st.sidebar.text_input("Enter YouTube video URL or ID:", value=f"https://www.youtube.com/watch?v={DEFAULT_VIDEO_ID}")

    video_id = extract_video_id(user_input)
    if video_id is None:
        st.error("Invalid YouTube video URL or ID. Please enter a valid URL or 11-character video ID.")
        return

    # Set up YouTube API client
    youtube = build("youtube", "v3", developerKey=API_KEY)

    with st.spinner("Fetching comments and analyzing..."):
        try:
            comments_data = get_comments(youtube, video_id, max_results=500)
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            return

        if not comments_data:
            st.error("No comments fetched for this video. Try a different one.")
            return

        df = pd.DataFrame(comments_data)
        df['clean_text'] = df['comment_text'].apply(clean_text)
        df['language'] = df['clean_text'].apply(detect_lang)
        df = df[df['language'] == 'en'].copy()
        df['lemmatized'] = df['clean_text'].apply(lemmatize_text)
        df['sentiment_score'] = df['lemmatized'].apply(lambda x: sid.polarity_scores(x)['compound'])
        df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)

    st.markdown(f"### Analyzing Video ID: `{video_id}`")
    st.metric("Total Comments Analyzed", len(df))

    # Sentiment distribution
    st.subheader("📈 Sentiment Distribution")
    fig = px.histogram(df, x='sentiment_label', color='sentiment_label',
                       category_orders={"sentiment_label": ["Positive", "Neutral", "Negative"]},
                       title="Sentiment Distribution")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # Word clouds per sentiment
    st.subheader("☁️ Word Clouds by Sentiment")
    col1, col2, col3 = st.columns(3)

    # Separate texts by sentiment
    positive_text = " ".join(df[df['sentiment_label']=="Positive"]['lemmatized'].tolist())
    neutral_text = " ".join(df[df['sentiment_label']=="Neutral"]['lemmatized'].tolist())
    negative_text = " ".join(df[df['sentiment_label']=="Negative"]['lemmatized'].tolist())

    wc_pos = generate_wordcloud(positive_text, 'Greens')
    wc_neu = generate_wordcloud(neutral_text, 'Greys')
    wc_neg = generate_wordcloud(negative_text, 'Reds')

    with col1:
        st.markdown("**Positive Sentiment (Green Shades)**")
        if wc_pos:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc_pos, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("No positive words available")

    with col2:
        st.markdown("**Neutral Sentiment (Grey Shades)**")
        if wc_neu:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc_neu, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("No neutral words available")

    with col3:
        st.markdown("**Negative Sentiment (Red Shades)**")
        if wc_neg:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc_neg, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("No negative words available")

    # Top entities display - simplistic approach using noun chunks
    st.subheader("🏷️ Top Entities Mentioned (Noun Phrases)")

    def extract_entities(text):
        doc = nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    df['entities'] = df['lemmatized'].apply(extract_entities)

    entity_list = list(itertools.chain.from_iterable(df['entities'].dropna().tolist()))
    counter = Counter(entity_list)
    top_n = 10
    top_entities = counter.most_common(top_n)

    if top_entities:
        entities_df = pd.DataFrame(top_entities, columns=['Entity', 'Count'])
        st.table(entities_df)

        st.subheader("📊 Column Chart for Top Entities")
        fig_entities = px.bar(entities_df, x='Entity', y='Count',
                              color='Count', color_continuous_scale='Viridis',
                              title='Top Entities Mentioned',
                              labels={'Count': 'Count', 'Entity': 'Entity'},
                              text='Count')
        fig_entities.update_traces(textposition='outside')
        fig_entities.update_layout(xaxis_tickangle=-45, yaxis_title='Count', coloraxis_showscale=False)
        st.plotly_chart(fig_entities, use_container_width=True)
    else:
        st.info("No entities found.")

    # Show sample comments with sentiment
    st.subheader("💬 Sample Comments")
    display_cols = ['comment_text', 'sentiment_label', 'entities']
    st.dataframe(df[display_cols].head(20).reset_index(drop=True))

    # Download analyzed data as CSV
    st.subheader("📥 Download Analyzed Comments (CSV)")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"youtube_comments_{video_id}.csv", "text/csv")

if __name__ == "__main__":
    main()
