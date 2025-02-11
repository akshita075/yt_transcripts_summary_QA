import os
import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
from urllib.parse import urlparse, parse_qs
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
import whisper
import yt_dlp
from pydub import AudioSegment

# --- CONFIGURE API KEYS ---
YOUTUBE_API_KEY = "AIzaSyBaNVUck5LpBp_t03g9SsxQgNG9e_KSA_o"  # Your YouTube API Key
GEMINI_API_KEY = "AIzaSyCqRjVXULLvSqVCoJYit6fOAXPWqLAQfUs"  # Your Google Gemini API Key

# --- Initialize Google Gemini API ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Initialize Sentence Transformer for Q/A ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit UI Layout ---
st.set_page_config(page_title="YouTube Video Assistant - RAG Enhanced", layout="wide")
st.title("🎥 YouTube Video Assistant - RAG Enhanced")
st.write("Process multiple YouTube videos for **transcription, summarization, and Q/A using RAG**.")

# ✅ Initialize session state
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "sentences" not in st.session_state:
    st.session_state.sentences = None

video_url = st.text_input("Enter YouTube Video URL:")

# --- FUNCTION: Extract and Clean YouTube Video URL ---
def clean_url(video_url):
    parsed_url = urlparse(video_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get("v", [None])[0]
    return (f"https://www.youtube.com/watch?v={video_id}", video_id) if video_id else (None, None)

# --- FUNCTION: Fetch Available Transcripts ---
def list_transcripts(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return [t.language_code for t in transcript_list]
    except Exception as e:
        return f"Error listing transcripts: {str(e)}"

# --- FUNCTION: Fetch Transcript with YouTubeTranscriptApi ---
@st.cache_data
def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to fetch manually created English transcript first
        for transcript in transcript_list:
            if transcript.language_code == "en":
                return " ".join([entry["text"] for entry in transcript.fetch()])

        # If no English, try auto-generated transcript
        for transcript in transcript_list:
            if transcript.is_generated:
                return " ".join([entry["text"] for entry in transcript.fetch()])

        return "⚠️ No English or auto-generated transcript found."
    
    except (TranscriptsDisabled, NoTranscriptFound):
        return transcribe_audio_whisper(f"https://www.youtube.com/watch?v={video_id}")

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# --- FUNCTION: Whisper AI Transcription ---
import pytube

import yt_dlp

def transcribe_audio_whisper(video_url):
    try:
        audio_file = "temp_audio.mp3"
        
        # Use yt-dlp to download audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_file,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Transcribe using Whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)

        os.remove(audio_file)  # Clean up
        return result["text"]

    except Exception as e:
        return f"Error with Whisper AI: {str(e)}"


# --- FUNCTION: Summarize Transcript using Google Gemini API ---
@st.cache_data
def summarize_transcript(transcript):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content(f"Summarize this:\n\n{transcript}")
        return response.text.strip()
    except Exception as e:
        return f"Error summarizing transcript with Gemini: {str(e)}"

# ✅ FUNCTION: Create FAISS Vector Index for RAG-based Q/A
@st.cache_data
def create_vector_index(transcript):
    sentences = transcript.split(". ")
    embeddings = embedding_model.encode(sentences)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    return index, sentences

# ✅ FUNCTION: Answer Question using RAG (Vector Search + Gemini)
def answer_question(transcript, question):
    try:
        if st.session_state.vector_index is None:
            st.session_state.vector_index, st.session_state.sentences = create_vector_index(transcript)

        index, sentences = st.session_state.vector_index, st.session_state.sentences

        question_embedding = embedding_model.encode([question]).astype("float32")
        _, indices = index.search(question_embedding, k=3)  # Get top 3 relevant sentences

        relevant_sentences = ". ".join([sentences[idx] for idx in indices[0]])

        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content(f"Context:\n{relevant_sentences}\n\nAnswer this question: {question}")

        return response.text.strip()
    except Exception as e:
        return f"Error answering question: {str(e)}"

# --- UI Tabs for Layout ---
tab1, tab2, tab3 = st.tabs(["📄 Transcription", "📑 Summarization", "❓ Q/A"])

with tab1:
    st.subheader("Transcript")
    
    if st.button("Generate Transcript") and video_url:
        _, video_id = clean_url(video_url)
        if video_id:
            st.session_state.transcript = fetch_transcript(video_id)

    if st.session_state.transcript:
        st.text_area("Transcript", st.session_state.transcript, height=300, key="transcript_tab1")

with tab2:
    st.subheader("Summary")
    
    if st.session_state.transcript:
        st.text_area("Transcript", st.session_state.transcript, height=150, disabled=True, key="transcript_tab2")

        if st.button("Summarize Content"):
            st.session_state.summary = summarize_transcript(st.session_state.transcript)
        
        if st.session_state.summary:
            st.text_area("Summary", st.session_state.summary, height=200, key="summary_tab")

with tab3:
    st.subheader("Ask a Question")

    if st.session_state.transcript:
        st.text_area("Transcript", st.session_state.transcript, height=150, disabled=True, key="transcript_tab3")

        question = st.text_input("Ask a question about the video:")
        if st.button("Get Answer"):
            answer = answer_question(st.session_state.transcript, question)
            st.write("Answer:", answer)
