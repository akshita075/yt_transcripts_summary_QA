import os
import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
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
from transformers import pipeline

# ✅ Ensure FFmpeg is installed
os.system("apt-get update && apt-get install -y ffmpeg libavcodec-extra")
AudioSegment.converter = which("ffmpeg")

# ✅ Load T5 Summarization Model
summarization_pipeline = pipeline("summarization", model="google/flan-t5-large")

# ✅ Initialize Sentence Transformer for Q/A
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
        return "**No transcript found via API. Switching to Whisper AI...**"

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# --- FUNCTION: Download Audio with yt-dlp ---
def download_audio(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'cookiefile': 'cookies.txt',  # ✅ Always use cookies
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
def transcribe_with_whisper(audio_path):
    try:
        st.write("🔄 Transcribing with Whisper AI...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        st.write("✅ Whisper AI Transcription Complete!")
        os.remove(audio_path)  # Cleanup
        return result["text"]
    except Exception as e:
        return f"Error with Whisper AI: {str(e)}"

# --- FUNCTION: Transcribe with AssemblyAI ---
def transcribe_with_assemblyai(audio_path):
    try:
        headers = {"authorization": "YOUR_ASSEMBLYAI_KEY"}
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
    except Exception as e:
        return f"Error with AssemblyAI: {str(e)}"

# --- FUNCTION: Summarize Transcript using T5 (instead of Gemini) ---
@st.cache_data
def summarize_transcript(transcript):
    if not transcript or transcript.strip() == "":
        return "⚠️ No transcript available to summarize."

    try:
        summary = summarization_pipeline(transcript, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"❌ Error summarizing transcript with T5: {str(e)}"

# ✅ FUNCTION: Create FAISS Vector Index for RAG-based Q/A
@st.cache_data
def create_vector_index(transcript):
    sentences = transcript.split(". ")
    embeddings = embedding_model.encode(sentences)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    return index, sentences

# ✅ FUNCTION: Answer Question using RAG (Vector Search + T5)
def answer_question(transcript, question):
    try:
        if st.session_state.vector_index is None:
            st.session_state.vector_index, st.session_state.sentences = create_vector_index(transcript)

        index, sentences = st.session_state.vector_index, st.session_state.sentences
        question_embedding = embedding_model.encode([question]).astype("float32")
        _, indices = index.search(question_embedding, k=3)
        relevant_sentences = ". ".join([sentences[idx] for idx in indices[0]])

        response = summarization_pipeline(f"Context: {relevant_sentences}\n\nQuestion: {question}\nAnswer:")
        return response[0]['summary_text']
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
            transcript = transcribe_with_whisper(audio_path) if audio_path else transcribe_with_assemblyai(audio_path)

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
