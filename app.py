import os
import streamlit as st
import yt_dlp
import whisper
import faiss
import numpy as np
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs

# ✅ API Keys (Keep these)
YOUTUBE_API_KEY = "AIzaSyBaNVUck5LpBp_t03g9SsxQgNG9e_KSA_o"
GEMINI_API_KEY = "AIzaSyCqRjVXULLvSqVCoJYit6fOAXPWqLAQfUs"
ASSEMBLYAI_API_KEY = "f8e218e5b7354f72ae11baeaff8d802f"

# ✅ Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Initialize FAISS for storing embeddings
def initialize_faiss_database(dim=384):
    return faiss.IndexFlatL2(dim)

def generate_embeddings(text):
    return np.array(embedding_model.encode([text]), dtype="float32")

# ✅ Streamlit UI Setup
st.set_page_config(page_title="YouTube Video Assistant", layout="wide")
st.title("🎥 YouTube Video Assistant - RAG Enhanced")

video_url = st.text_input("Enter YouTube Video URL:")

# ✅ Extract Video ID
def clean_url(video_url):
    parsed_url = urlparse(video_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get("v", [None])[0]
    return (f"https://www.youtube.com/watch?v={video_id}", video_id) if video_id else (None, None)

# ✅ Fetch YouTube Transcript
def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            if transcript.language_code == "en":
                return " ".join([entry["text"] for entry in transcript.fetch()])
        return "**No YouTube transcript found. Switching to Whisper AI...**"
    except (TranscriptsDisabled, NoTranscriptFound):
        return "**No transcript found via API. Trying Whisper AI...**"

# ✅ Download Audio
def download_audio(video_url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            file_name = ydl.prepare_filename(info).replace(".webm", ".mp3")
        return file_name
    except Exception as e:
        return f"Error downloading audio: {str(e)}"

# ✅ Transcribe with Whisper
def transcribe_audio_whisper(audio_path):
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error with Whisper AI: {str(e)}"

# ✅ Summarize Transcript
def summarize_transcript(transcript):
    return "Summary: " + transcript[:300] + "..."

# ✅ FAISS-based Q/A
def answer_question(transcript, question):
    index = initialize_faiss_database()
    embedding = generate_embeddings(transcript)
    index.add(embedding)

    question_embedding = generate_embeddings(question)
    _, indices = index.search(question_embedding, k=1)

    relevant_text = transcript
    return f"Q: {question}\nA: {relevant_text}"

# ✅ Streamlit UI Tabs
tab1, tab2, tab3 = st.tabs(["📄 Transcription", "📑 Summarization", "❓ Q/A"])

with tab1:
    st.subheader("Transcript")
    if st.button("Generate Transcript") and video_url:
        _, video_id = clean_url(video_url)
        transcript = fetch_transcript(video_id)

        if "**No transcript found**" in transcript:
            audio_path = download_audio(video_url)
            transcript = transcribe_audio_whisper(audio_path)

        st.session_state.transcript = transcript
        st.text_area("Transcript", transcript, height=300)

with tab2:
    st.subheader("Summary")
    if st.button("Summarize Content") and "transcript" in st.session_state:
        st.session_state.summary = summarize_transcript(st.session_state.transcript)
        st.text_area("Summary", st.session_state.summary, height=200)

with tab3:
    st.subheader("Ask a Question")
    question = st.text_input("Ask a question about the video:")
    if st.button("Get Answer"):
        answer = answer_question(st.session_state.transcript, question)
        st.write("Answer:", answer)
