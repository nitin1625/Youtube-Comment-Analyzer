import googleapiclient.discovery 
import re
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

def get_youtube_comments(video_url, api_key):
    # Extract video ID from the URL
    video_id = video_url.split('v=')[1].split('&')[0]

    # Initialize the YouTube API client
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    

    # Fetch comments
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=300  # Number of comments to retrieve
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
        comments.append(comment)

    return comments

# Function to preprocess comments
def preprocess_comments(comments):
    cleaned_comments = []
    for comment in comments:
        # Remove URLs, mentions, and special characters
        comment = re.sub(r"http\S+|www\S+|https\S+|@\S+", '', comment, flags=re.MULTILINE)
        comment = re.sub(r'\s+', ' ', comment)  # Remove extra spaces
        comment = re.sub(r'[^\w\s]', '', comment)  # Remove punctuation
        cleaned_comments.append(comment.strip())
    return cleaned_comments

# Function to interact with Gemini API
def analyze_sentiments(sentiments):
    positive_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'POSITIVE')
    negative_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'NEGATIVE')

    total_comments = len(sentiments)

    positive_percentage = (positive_count / total_comments) * 100
    negative_percentage = (negative_count / total_comments) * 100

    overall_sentiment = "POSITIVE" if positive_count >= negative_count else "NEGATIVE"

    analysis = {
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        "overall_sentiment": overall_sentiment,
        "liked_by_people": "Yes" if overall_sentiment == "POSITIVE" else "No"
    }

    return analysis

def analyze_comments(comments):
    nlp = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = nlp(comments)
    positive_comments = [comment for comment, sentiment in zip(comments, sentiments) if sentiment['label'] == 'POSITIVE']
    negative_comments = [comment for comment, sentiment in zip(comments, sentiments) if sentiment['label'] == 'NEGATIVE']

    summary = generate_summary(comments)
    likes = extract_likes(positive_comments)
    improvements = extract_improvements(negative_comments)
    dislikes = extract_dislikes(negative_comments)
    sentiment_analysis=analyze_sentiments(sentiments)

    analysis = {
         "summary": summary,
        "likes": likes,
        "improvements": improvements,
        "dislikes": dislikes,
        "sentiment_analysis": sentiment_analysis
    }

    print(analysis)

    return analysis

# Helper functions to generate insights
def generate_summary(comments):
    # Use an LLM to summarize the topic of the video
    prompt = "Summarize the main topic of the following comments in less than 20 words:\n" + "\n".join(comments)
    summary = llm.invoke(prompt)
    return summary.content

def extract_likes(comments):
    prompt = "What do people like about this video based on these comments?Explain under 50 words in pointwise manner \n" + "\n".join(comments)
    likes = llm.invoke(prompt)
    return likes.content

def extract_improvements(comments):
    prompt = "What improvements do people suggest for this video based on these comments?Explain under 50 words in pointwise manner\n" + "\n".join(comments)
    improvements = llm.invoke(prompt)
    return improvements.content

def extract_dislikes(comments):
    prompt = "What do people dislike about this video based on these comments?Explain under 50 words in pointwise manner\n" + "\n".join(comments)
    dislikes = llm.invoke(prompt)
    return dislikes.content

