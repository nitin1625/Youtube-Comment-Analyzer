import streamlit as st
from dotenv import load_dotenv
import os
from back import get_youtube_comments, preprocess_comments, analyze_comments

# Load environment variables
load_dotenv()
youtube_api_key = os.getenv('YOUTUBE_API_KEY')


def main():
    st.title("YouTube Comments Analyzer")
    
    video_url = st.text_input("Enter YouTube Video URL")
    
    if st.button("Analyze"):
        if not video_url:
            st.error("No video URL provided")
        else:
            try:
                comments = get_youtube_comments(video_url, youtube_api_key)
                if not comments:
                    st.error("No comments found")
                else:
                    cleaned_comments = preprocess_comments(comments)
                    analysis = analyze_comments(cleaned_comments)
                    st.write("Analysis Results")
                    st.json(analysis)
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
