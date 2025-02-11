import pandas as pd
import chromadb  # Use FAISS if preferred
from sentence_transformers import SentenceTransformer

# Load YouTube transcripts
df = pd.read_csv("youtube_transcripts.csv")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # This will store data persistently
collection = chroma_client.get_or_create_collection("youtube_transcripts")

# Process transcripts
for index, row in df.iterrows():
    video_id = row["video_id"]
    transcript = row["transcript"]

    # Split transcript into smaller chunks (e.g., sentences)
    sentences = transcript.split(". ")
    embeddings = embedding_model.encode(sentences).tolist()

    # Store in vector DB
    for sentence, embedding in zip(sentences, embeddings):
        collection.add(
            ids=[f"{video_id}_{index}"], 
            embeddings=[embedding], 
            metadatas=[{"video_id": video_id, "text": sentence}]
        )

print("✅ Transcripts stored in ChromaDB!")
