import os
import sys
import progressbar

parent_directory = os.path.abspath('..')

sys.path.append(parent_directory)


from vts import (
    audio_processing,
    metadata_generation,
    segment_analysis,
    topic_modeling,
    transcription,
    utils,
)

def save_transcription(transcription, project_path):
    transcription_path = os.path.join(project_path, "transcription.json")
    with open(transcription_path, "w") as f:
        json.dump(transcription, f, indent=2)
    print(f"Transcription saved to: {transcription_path}")


def save_transcript(transcript, project_path):
    transcript_path = os.path.join(project_path, "transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)
    print(f"Transcript saved to: {transcript_path}")


def load_transcript(transcript_path):
    with open(transcript_path, "r") as f:
        return json.load(f)


def generate_metadata(segments, lda_model):
    print("Generating metadata for segments...")
    metadata = []
    for i, segment in enumerate(progressbar.progressbar(segments)):
        segment_metadata = {
            "segment_id": i + 1,
            "start_time": segment["start"],
            "end_time": segment["end"],
            "duration": segment["end"] - segment["start"],
            "dominant_topic": segment["topic"],
            "top_keywords": [
                word for word, _ in lda_model.show_topic(segment["topic"], topn=5)
            ],
            "transcript": segment["content"],
        }
        metadata.append(segment_metadata)
    print("Metadata generation complete.")
    return metadata
