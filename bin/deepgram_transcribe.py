import os
import argparse
import time
import subprocess
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq
import sys

load_dotenv()

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
API_CALL_DELAY = 1  # Delay between API calls in seconds

def preprocess_audio(input_file, output_file):
    """Convert, apply HPF, and normalize audio file to M4A format."""
    command = [
        "ffmpeg", "-i", input_file,
        "-af", "highpass=f=60, acompressor=threshold=-12dB:ratio=4:attack=5:release=50, loudnorm",
        "-ar", "16000", "-ac", "1",
        "-c:a", "aac", "-b:a", "128k",  # Use AAC codec for M4A
        "-progress", "pipe:1",  # Output progress information to stdout
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
                    "mimetype": "audio/mp4"  # Changed to audio/mp4 for M4A files
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

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files in a folder using Deepgram or Groq API.")
    parser.add_argument("folder_path", help="Path to the folder containing audio files")
    parser.add_argument("--api", choices=["deepgram", "groq"], default="deepgram", help="Choose API: deepgram or groq")
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY, help="Delay between API calls in seconds")
    args = parser.parse_args()

    try:
        # Initialize both clients
        deepgram_key = os.getenv("DG_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not deepgram_key:
            raise ValueError("DG_API_KEY environment variable is not set")
        if not groq_key and args.api == "groq":
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

        for filename in os.listdir(args.folder_path):
            if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.opus', '.m4a')):
                file_path = os.path.join(args.folder_path, filename)
                print(f"Processing {filename}...")
                
                try:
                    # Preprocess the audio file
                    preprocessed_file = os.path.join(args.folder_path, f"preprocessed_{filename}.m4a")
                    preprocess_audio(file_path, preprocessed_file)
                    
                    # Check file size after preprocessing
                    file_size = os.path.getsize(preprocessed_file)
                    
                    if file_size > MAX_FILE_SIZE or args.api == "deepgram":
                        response = transcribe_file_deepgram(deepgram_client, preprocessed_file, deepgram_options)
                        transcription = response.to_json(indent=4)
                        api_used = "deepgram"
                    else:  # Groq
                        transcription = transcribe_file_groq(groq_client, preprocessed_file)
                        api_used = "groq"
                    
                    # Save the transcription to a text file
                    output_filename = os.path.splitext(filename)[0] + f"_{api_used}_transcription.txt"
                    output_path = os.path.join(args.folder_path, output_filename)
                    with open(output_path, "w") as output_file:
                        output_file.write(transcription)
                    
                    print(f"Transcription saved to {output_filename} using {api_used} API")
                    
                    # Remove the preprocessed file
                    os.remove(preprocessed_file)
                    
                    # Sleep to respect rate limits
                    time.sleep(args.delay)
                
                except (DeepgramError, Exception) as e:
                    print(f"Error while processing {filename}: {str(e)}")
                    if os.path.exists(preprocessed_file):
                        os.remove(preprocessed_file)

    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
