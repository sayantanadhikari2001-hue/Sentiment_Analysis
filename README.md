🎯 YouTube Video Sentiment Analysis with Word Clouds

A Streamlit-based interactive dashboard that analyzes YouTube video comments using NLP techniques such as sentiment analysis, lemmatization, entity extraction, and word cloud visualization.

📌 Project Overview

This project allows users to enter any YouTube video URL or ID, fetches up to 500 comments using the YouTube Data API, processes the textual data, and generates multiple visual insights:

📈 Sentiment distribution (Positive / Neutral / Negative)

☁️ Word clouds for each sentiment type

🏷️ Top entities (noun phrases) extracted using spaCy

💬 Sample processed comments with sentiment labels

📥 Downloadable CSV of all analyzed comments

The entire application runs on an interactive Streamlit dashboard.

🚀 Features
✔️ YouTube Data Fetching

Fetches comments via the YouTube Data API v3 using a user-provided URL/ID.

✔️ Text Cleaning & Preprocessing

Emoji removal

URL removal

Removing special characters

Lowercasing

Language detection (keeps only English comments)

Lemmatization using spaCy

✔️ Sentiment Analysis

Uses VADER (NLTK) to compute compound scores and classify comments into:

Positive

Neutral

Negative

✔️ Word Cloud Generation

Generates word clouds using distinct color themes:

Sentiment	Color Scheme
Positive	Green shades
Neutral	Grey shades
Negative	Red shades
✔️ Entity Extraction

Extracts noun phrases using spaCy’s noun_chunks.
Displays:

Top 10 entities

Interactive Plotly bar chart

✔️ Export Option

Download processed comments as a CSV file.

🛠️ Tech Stack
Category	Tools / Libraries
Frontend	Streamlit
Backend	Python
API	YouTube Data API v3
NLP	spaCy, NLTK, langdetect, emoji
Visualization	Plotly, Matplotlib, WordCloud
Data Handling	Pandas, NumPy
📂 Project Structure
.
├── app.py                   # Main Streamlit Application
├── README.md                # Project Documentation
├── requirements.txt         # Python Dependencies
└── assets/                  # (Optional) Images, screenshots

🔑 Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/your-username/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis

2️⃣ Install Dependencies

Create a virtual environment (recommended), then:

pip install -r requirements.txt

3️⃣ Add YouTube API Key

Inside app.py, replace this placeholder with your API key:

API_KEY = "YOUR_API_KEY"


Get your API Key from
👉 https://console.cloud.google.com/apis/library/youtube.googleapis.com

4️⃣ Run the Application
streamlit run app.py

🧪 How to Use the App

Open the Streamlit UI.

Enter any YouTube video URL or ID in the sidebar.

Wait for the app to fetch and analyze comments.

Explore:

Sentiment histogram

Word clouds

Top entities bar chart

Sample comments table

Download the results as CSV (optional).

📦 Requirements

Your requirements.txt (updated):

pandas
streamlit>=1.18
plotly
wordcloud
matplotlib
nltk
spacy
gensim
langdetect
emoji
openpyxl
numpy
seaborn
google-api-python-client

📊 Screenshots (Optional)

Add screenshots of your dashboard here.

🌱 Future Enhancements

Add reply comment analysis

Multi-language sentiment support

Emotion classification (joy, anger, sadness, etc.)

Topic modeling using LDA

Export charts as images

🤝 Contribution

Contributions are welcome!
Feel free to open an issue or submit a pull request.

📜 License

Distributed under the MIT License.
