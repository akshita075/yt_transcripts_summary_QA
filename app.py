import streamlit as st  # ✅ Import Streamlit first
st.set_page_config(page_title="YouTube Video Assistant", layout="wide")  # ✅ MUST be first Streamlit command

import os
import streamlit as st  # ✅ Import first
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
from urllib.parse import urlparse, parse_qs
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import yt_dlp
import whisper
import time
from pydub import AudioSegment
from pydub.utils import which

# ✅ **THIS MUST BE THE FIRST STREAMLIT COMMAND**
st.set_page_config(page_title="YouTube Video Assistant", layout="wide")


# ✅ Set FFmpeg path for pydub
AudioSegment.converter = which("ffmpeg")

# ✅ Initialize API Keys
YOUTUBE_API_KEY = "AIzaSyBaNVUck5LpBp_t03g9SsxQgNG9e_KSA_o"
GEMINI_API_KEY = "AIzaSyCqRjVXULLvSqVCoJYit6fOAXPWqLAQfUs"
ASSEMBLYAI_API_KEY = "f8e218e5b7354f72ae11baeaff8d802f"

# ✅ Initialize Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)


# --- Initialize Sentence Transformer for Faster Q/A ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit UI Layout ---
st.set_page_config(page_title="YouTube Video Assistant", layout="wide")

st.title("🎥 YouTube Video Assistant")
st.write("Process a YouTube video for **transcription, summarization, and Q/A**.")

video_url = st.text_input("Enter YouTube Video URL:")

# --- FUNCTION: Extract and Clean YouTube Video URL ---
def clean_url(video_url):
    parsed_url = urlparse(video_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get("v", [None])[0]

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}", video_id
    elif "youtu.be" in video_url:
        video_id = parsed_url.path.lstrip("/")
        return f"https://youtu.be/{video_id}", video_id
    else:
        return None, None

# --- FUNCTION: Fetch Captions using YouTube API ---
@st.cache_data
def get_video_captions(video_url):
    _, video_id = clean_url(video_url)
    if not video_id:
        return "Invalid YouTube URL!", None

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.captions().list(part="snippet", videoId=video_id)
        response = request.execute()

        if "items" in response and response["items"]:
            captions = [item["snippet"]["language"] for item in response["items"]]
            return f"Available captions: {', '.join(captions)}", video_id
        else:
            return "No captions available.", video_id

    except Exception as e:
        return f"Error fetching captions: {str(e)}", None

# --- FUNCTION: Fetch Transcript using YouTubeTranscriptApi ---
@st.cache_data
def fetch_transcript(video_id, language="en"):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        full_transcript = " ".join([entry["text"] for entry in transcript])
        return full_transcript
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# --- FUNCTION: Summarize Transcript using Google Gemini API ---
@st.cache_data
def summarize_transcript(transcript):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content(f"Summarize this:\n\n{transcript}")
        return response.text.strip()
    except Exception as e:
        return f"Error summarizing transcript with Gemini: {str(e)}"

# --- FUNCTION: Generate Vector Index for Fast Q/A ---
@st.cache_data
def create_vector_index(transcript):
    sentences = transcript.split(". ")
    sentence_embeddings = embedding_model.encode(sentences)

    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(np.array(sentence_embeddings).astype("float32"))

    return index, sentences

# --- FUNCTION: Answer Questions using Gemini API + Vector Search ---
def answer_question(transcript, question):
    try:
        index, sentences = create_vector_index(transcript)

        question_embedding = embedding_model.encode([question]).astype("float32")
        _, indices = index.search(question_embedding, k=3)  # Get top 3 most relevant sentences

        relevant_sentences = ". ".join([sentences[idx] for idx in indices[0]])

        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content(f"Based on this transcript:\n\n{relevant_sentences}\n\nAnswer this question: {question}")
        return response.text.strip()
    except Exception as e:
        return f"Error answering question: {str(e)}"

# --- FUNCTION: Download Audio from YouTube ---
@st.cache_data
def download_audio(video_url):
    try:
        cleaned_url, _ = clean_url(video_url)
        if not cleaned_url:
            return "Error: Invalid YouTube URL."

        command = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",
            "-o", "audio.%(ext)s",
            cleaned_url
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            return "audio.mp3"
        else:
            return f"Error downloading audio: {result.stderr}"

    except Exception as e:
        return f"Error downloading audio: {str(e)}"

# --- FUNCTION: Transcribe Audio using Whisper ---
@st.cache_data
def transcribe_audio(audio_file):
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

# --- UI Tabs for Better Layout ---
tab1, tab2, tab3 = st.tabs(["📄 Transcription", "📑 Summarization", "❓ Q/A"])

if "transcript" not in st.session_state:
    st.session_state.transcript = None

with tab1:
    if st.button("Generate Transcript"):
        if video_url:
            captions, video_id = get_video_captions(video_url)
            if "Available captions" in captions:
                st.session_state.transcript = fetch_transcript(video_id)
            else:
                st.write("No captions available. Downloading audio...")
                audio_file = download_audio(video_url)
                if "Error" not in audio_file:
                    st.session_state.transcript = transcribe_audio(audio_file)
                else:
                    st.session_state.transcript = audio_file  # Store error message

            st.text_area("Transcript", st.session_state.transcript, height=300)
        else:
            st.write("⚠️ Please enter a valid YouTube URL.")

with tab2:
    if st.button("Summarize Content"):
        if st.session_state.transcript:
            summary = summarize_transcript(st.session_state.transcript)
            st.text_area("Summary", summary, height=200)
        else:
            st.write("⚠️ Generate transcript first.")

with tab3:
    question = st.text_input("Ask a question about the video:")
    if st.button("Get Answer"):
        if st.session_state.transcript:
            answer = answer_question(st.session_state.transcript, question)
            st.write("Answer:")
            st.write(answer)
        else:
            st.write("⚠️ Generate transcript first.")
