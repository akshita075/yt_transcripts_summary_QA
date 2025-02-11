from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd

# List of YouTube video IDs
video_ids = [
    '6Iu45VZGQDk',
    '7S_tz1z_5bA',
    'vBURTt97EkA',
    'LOfGJcVnvAk',
    'WjwEh15M5Rw',
    'SaCYkPD4_K0',
    '4Cr0OxXU7jY',
    '_uQrJ0TkZlc',
    '_V8eKsto3Ug',
    # Add more video IDs as needed
]

transcripts = []

for video_id in video_ids:
    try:
        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine transcript text
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        transcripts.append({'video_id': video_id, 'transcript': transcript_text})
    except Exception as e:
        print(f"Could not retrieve transcript for {video_id}: {e}")

# Create a DataFrame and save to CSV
df = pd.DataFrame(transcripts)
df.to_csv('youtube_transcripts.csv', index=False)



AIzaSyBaNVUck5LpBp_t03g9SsxQgNG9e_KSA_o
AIzaSyCqRjVXULLvSqVCoJYit6fOAXPWqLAQfUs