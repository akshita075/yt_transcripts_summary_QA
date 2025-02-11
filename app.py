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

# --- CONFIGURE API KEYS ---
YOUTUBE_API_KEY = "AIzaSyBaNVUck5LpBp_t03g9SsxQgNG9e_KSA_o"  # Replace with your YouTube API key
GEMINI_API_KEY = "AIzaSyCqRjVXULLvSqVCoJYit6fOAXPWqLAQfUs"  # Replace with your Google Gemini API key
ASSEMBLYAI_API_KEY = "f8e218e5b7354f72ae11baeaff8d802f"  # Your AssemblyAI API key

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

# --- FUNCTION: Fetch Transcript using YouTubeTranscriptAPI ---
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

        return "**No transcript found via API. Trying external transcription...**"
    
    except (TranscriptsDisabled, NoTranscriptFound):
        return transcribe_audio_assemblyai(f"https://www.youtube.com/watch?v={video_id}")

    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# --- FUNCTION: Transcribe Audio using AssemblyAI ---
def transcribe_audio_assemblyai(video_url):
    try:
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        response = requests.post("https://api.assemblyai.com/v2/transcript",
                                 json={"audio_url": video_url},
                                 headers=headers)
        data = response.json()
        transcript_id = data.get("id")

        # Wait for the transcript to be processed
        while True:
            transcript_response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                                               headers=headers)
            transcript_data = transcript_response.json()

            if transcript_data.get("status") == "completed":
                return transcript_data["text"]
            elif transcript_data.get("status") == "failed":
                return "❌ Failed to transcribe the audio."
    
    except Exception as e:
        return f"Error with external transcription: {str(e)}"

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
        response 
