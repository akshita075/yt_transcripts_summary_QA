from youtube_transcript_api import YouTubeTranscriptApi
import os

# Directory to save transcripts
output_dir = "transcripts"
os.makedirs(output_dir, exist_ok=True)

# List of YouTube video IDs (DBMS, Python, R, Software Engineering, Agile, etc.)
video_ids = [
    '6Iu45VZGQDk', '7S_tz1z_5bA', 'vBURTt97EkA', 'LOfGJcVnvAk',
    'WjwEh15M5Rw', 'SaCYkPD4_K0', '4Cr0OxXU7jY', '_uQrJ0TkZlc', '_V8eKsto3Ug'
]

for video_id in video_ids:
    try:
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = '\n'.join([entry['text'] for entry in transcript])

        # Save transcript as a text file
        file_path = os.path.join(output_dir, f"{video_id}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(transcript_text)

        print(f"✅ Saved transcript for {video_id}")

    except Exception as e:
        print(f"❌ Could not retrieve transcript for {video_id}: {e}")
