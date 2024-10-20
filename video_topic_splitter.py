import os
import argparse
import time
import subprocess
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq
import sys
from moviepy.editor import VideoFileClip
from datetime import datetime

load_dotenv()

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
API_CALL_DELAY = 1  # Delay between API calls in seconds

def create_project_folder(input_video, base_output_dir):
    """Create a new project folder based on the input video name and timestamp."""
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"{base_name}_{timestamp}"
    project_path = os.path.join(base_output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path

def extract_audio(video_path, output_path):
    """Extract audio from the input video file."""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()

def preprocess_audio(input_file, output_file):
    """Convert, apply HPF, and normalize audio file to M4A format."""
    command = [
        "ffmpeg", "-i", input_file,
        "-af", "highpass=f=60, acompressor=threshold=-12dB:ratio=4:attack=5:release=50, loudnorm",
        "-ar", "16000", "-ac", "1",
        "-c:a", "aac", "-b:a", "128k",
        "-progress", "pipe:1",
        output_file
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    duration = None
    for line in process.stdout:
        if line.startswith("Duration: "):
            duration = line.split()[1].strip(',')
        if line.startswith("out_time="):
            current_time = line.split('=')[1].strip()
            if duration:
                progress = (time_to_seconds(current_time) / time_to_seconds(duration)) * 100
                print_progress_bar(progress)
    
    process.wait()
    print()  # New line after progress bar

def time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def print_progress_bar(progress):
    bar_length = 50
    filled_length = int(bar_length * progress // 100)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\rProgress: [{bar}] {progress:.1f}%')
    sys.stdout.flush()

def transcribe_file_deepgram(client, file_path, options, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                payload: FileSource = {
                    "buffer": buffer_data,
                    "mimetype": "audio/mp4"
                }
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            return response
        except DeepgramError as e:
            if attempt < max_retries - 1:
                print(f"API call failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise e
        except Exception as e:
            raise e

def transcribe_file_groq(client, file_path, model="whisper-large-v3", language="en"):
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model=model,
            language=language,
            response_format="text"
        )
    return transcription

def process_video(video_path, project_path, api="deepgram"):
    """Process the input video: extract audio, preprocess, and transcribe."""
    # Create subdirectories
    audio_dir = os.path.join(project_path, "audio")
    transcript_dir = os.path.join(project_path, "transcript")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)
    
    # Extract audio
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.wav")
    extract_audio(video_path, raw_audio_path)
    
    # Preprocess audio
    preprocessed_audio_path = os.path.join(audio_dir, "preprocessed_audio.m4a")
    preprocess_audio(raw_audio_path, preprocessed_audio_path)
    
    # Initialize clients
    deepgram_key = os.getenv("DG_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not deepgram_key:
        raise ValueError("DG_API_KEY environment variable is not set")
    if not groq_key and api == "groq":
        raise ValueError("GROQ_API_KEY environment variable is not set")

    deepgram_client = DeepgramClient(deepgram_key)
    groq_client = Groq(api_key=groq_key) if groq_key else None

    deepgram_options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        language="en",
        punctuate=True,
        utterances=True,
        diarize=True,
    )

    # Transcribe
    file_size = os.path.getsize(preprocessed_audio_path)
    if file_size > MAX_FILE_SIZE or api == "deepgram":
        response = transcribe_file_deepgram(deepgram_client, preprocessed_audio_path, deepgram_options)
        transcription = response.to_json(indent=4)
        api_used = "deepgram"
    else:  # Groq
        transcription = transcribe_file_groq(groq_client, preprocessed_audio_path)
        api_used = "groq"
    
    # Save transcription
    transcript_path = os.path.join(transcript_dir, f"transcription_{api_used}.json")
    with open(transcript_path, 'w') as f:
        f.write(transcription)
    
    # Clean up
    os.remove(raw_audio_path)
    os.remove(preprocessed_audio_path)
    
    return transcription, api_used

def main():
    parser = argparse.ArgumentParser(description="Split video based on topics")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output", default=os.getcwd(), help="Base output directory for project folders")
    parser.add_argument("--api", choices=["deepgram", "groq"], default="deepgram", help="Choose API: deepgram or groq")
    args = parser.parse_args()

    project_path = create_project_folder(args.input, args.output)
    transcription, api_used = process_video(args.input, project_path, args.api)
    
    print(f"Processing complete. Project folder: {project_path}")
    print(f"Transcription saved using {api_used} API in: {os.path.join(project_path, 'transcript', f'transcription_{api_used}.json')}")

if __name__ == "__main__":
    main()