import os
import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
from urllib.parse import urlparse, parse_qs
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# ✅ Import pydub for audio processing
from pydub import AudioSegment
from pydub.utils import which

# ✅ Ensure FFmpeg is installed
os.system("apt-get update && apt-get install -y ffmpeg libavcodec-extra")

# ✅ Set FFmpeg path for pydub
AudioSegment.converter = which("ffmpeg")

import os
import streamlit as st
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

import os

# Install FFmpeg if not found
if os.system("ffmpeg -version") != 0:
    os.system("apt-get update && apt-get install -y ffmpeg")

import os
from pydub.utils import which

# Set FFmpeg path manually
ffmpeg_path = os.path.abspath("ffmpeg-7.1-essential-builds/ffmpeg")
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Ensure pydub uses the correct FFmpeg
AudioSegment.converter = which("ffmpeg")


# --- CONFIGURE API KEYS ---
YOUTUBE_API_KEY = "AIzaSyBaNVUck5LpBp_t03g9SsxQgNG9e_KSA_o"
GEMINI_API_KEY = "AIzaSyCqRjVXULLvSqVCoJYit6fOAXPWqLAQfUs"
ASSEMBLYAI_API_KEY = "f8e218e5b7354f72ae11baeaff8d802f"

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

# --- FUNCTION: Fetch YouTube Transcript ---
@st.cache_data
def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        for transcript in transcript_list:
            if transcript.language_code == "en":
                return " ".join([entry["text"] for entry in transcript.fetch()])

        for transcript in transcript_list:
            if transcript.is_generated:
                return " ".join([entry["text"] for entry in transcript.fetch()])

        return "**No YouTube transcript. Trying Whisper AI...**"
    
    except (TranscriptsDisabled, NoTranscriptFound):
        st.write("📢 No transcript found via YouTube API, switching to Whisper AI...")
        return transcribe_audio_whisper(f"https://www.youtube.com/watch?v={video_id}")

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"


# --- FUNCTION: Download Audio with yt-dlp ---
def download_audio(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return "temp_audio.mp3"
    except Exception as e:
        return f"Error downloading audio: {str(e)}"

# --- FUNCTION: Transcribe with Whisper ---
import whisper
import yt_dlp

def transcribe_audio_whisper(video_url):
    try:
        st.write("🔄 Downloading audio for Whisper AI...")
        
        # Download YouTube audio using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': "temp_audio.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            file_name = ydl.prepare_filename(info_dict).replace('.webm', '.mp3')

        st.write(f"🎵 Audio downloaded: {file_name}")

        # Load Whisper model and transcribe
        model = whisper.load_model("base")
        result = model.transcribe(file_name)

        st.write("✅ Whisper AI Transcription Complete!")

        # Cleanup
        os.remove(file_name)

        return result["text"]

    except Exception as e:
        return f"Error with Whisper AI: {str(e)}"


# --- FUNCTION: Transcribe with AssemblyAI ---
def transcribe_with_assemblyai(audio_path):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(audio_path, "rb") as f:
        response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, files={"file": f})
    if response.status_code != 200:
        return f"Error uploading file: {response.text}"

    audio_url = response.json().get("upload_url")
    response = requests.post("https://api.assemblyai.com/v2/transcript", json={"audio_url": audio_url}, headers=headers)
    transcript_id = response.json().get("id")

    while True:
        transcript_response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        transcript_data = transcript_response.json()
        if transcript_data.get("status") == "completed":
            return transcript_data["text"]
        elif transcript_data.get("status") == "failed":
            return "❌ Failed to transcribe the audio."
        time.sleep(5)

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
        _, indices = index.search(question_embedding, k=3)

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
        transcript = fetch_transcript(video_id)

        if "**No transcript found via API**" in transcript:
            audio_path = download_audio(video_url)
            transcript = transcribe_with_whisper(audio_path)
        
        if "Error" in transcript:
            transcript = transcribe_with_assemblyai(audio_path)

        st.session_state.transcript = transcript
        st.text_area("Transcript", transcript, height=300)

with tab2:
    st.subheader("Summary")
    if st.button("Summarize Content") and st.session_state.transcript:
        st.session_state.summary = summarize_transcript(st.session_state.transcript)
        st.text_area("Summary", st.session_state.summary, height=200)

with tab3:
    st.subheader("Ask a Question")
    question = st.text_input("Ask a question about the video:")
    if st.button("Get Answer"):
        st.write("Answer:", answer_question(st.session_state.transcript, question))
