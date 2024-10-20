#!/usr/bin/env python
import os
import shutil
import subprocess
import argparse
import psutil
import tempfile
import time
from unsilence import Unsilence
from pydub import AudioSegment  # Import AudioSegment

# Configuration constants
MAX_RETRIES = 3  # Maximum number of retry attempts for Whisper transcription
RETRY_DELAY = 5  # Delay in seconds between retry attempts

def run_whisper(input_file, output_file, verbose=False):
    """
    Transcribe audio using the Whisper model.
    
    Args:
    input_file (str): Path to the input audio file.
    output_file (str): Path to save the transcription output.
    verbose (bool): If True, print detailed progress information.
    
    Returns:
    None
    """
    for attempt in range(MAX_RETRIES):
        try:
            if verbose:
                print(f"Attempt {attempt + 1} to transcribe using Whisper")
            subprocess.run(["whisper.cpp", input_file, "-o", output_file], check=True)
            return  # Success, exit the function
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Whisper transcription failed: {e}")
            if attempt < MAX_RETRIES - 1:
                if verbose:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to transcribe after {MAX_RETRIES} attempts. Skipping transcription.")
                # Create an empty file to indicate transcription was attempted but failed
                with open(output_file, 'w') as f:
                    f.write("Transcription failed after multiple attempts.")

def run_sonic_annotator(input_file, output_dir):
    """
    Analyze audio features using Sonic Annotator.
    
    Args:
    input_file (str): Path to the input audio file.
    output_dir (str): Directory to save the analysis output.
    
    Returns:
    None
    """
    subprocess.run(["sonic-annotator", "-d", "vamp:qm-vamp-plugins:qm-tempotracker:tempo", 
                    "-d", "vamp:qm-vamp-plugins:qm-keydetector:key", 
                    "-w", "csv", "--csv-force", 
                    "-o", output_dir, input_file], check=True)

def transcode(input_file, output_file):
    """
    Transcode video to MP4 format with specific encoding options.
    
    Args:
    input_file (str): Path to the input video file.
    output_file (str): Path to save the transcoded video.
    
    Returns:
    None
    """
    subprocess.run(["ffmpeg", "-i", input_file, "-c:v", "libx264", "-crf", "23", 
                    "-c:a", "aac", "-b:a", "128k", output_file], check=True)

def extract_audio(input_file, output_file):
    """
    Extract audio from a video file as WAV.
    
    Args:
    input_file (str): Path to the input video file.
    output_file (str): Path to save the extracted audio.
    
    Returns:
    None
    """
    subprocess.run(["ffmpeg", "-i", input_file, "-vn", "-acodec", "pcm_s16le", 
                    "-ar", "44100", "-ac", "2", output_file], check=True)

def estimate_silence(input_file):
    """
    Estimate the percentage of silence in an audio file.
    
    Args:
    input_file (str): Path to the input audio file.
    
    Returns:
    float: Percentage of silence in the audio.
    """
    u = Unsilence(input_file)
    u.detect_silence()
    
    audio = AudioSegment.from_file(input_file)  # Load audio with pydub
    total_duration = audio.duration_seconds  # Get duration in seconds
    
    silent_duration = sum(interval[1] - interval[0] for interval in u.silent_intervals)
    silence_percentage = (silent_duration / total_duration) * 100
    return silence_percentage

def unsilence_audio(input_file, output_file, audible_speed=1, silent_speed=8):
    """
    Remove silence from an audio/video file if silence percentage is above 10%.
    
    Args:
    input_file (str): Path to the input audio/video file.
    output_file (str): Path to save the unsilenced audio/video.
    audible_speed (int): Speed for audible segments (default: 1).
    silent_speed (int): Speed for silent segments (default: 8).
    
    Returns:
    None
    """
    if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Extract audio from video
        temp_dir = tempfile.mkdtemp()
        audio_file = os.path.join(temp_dir, "temp_audio.wav")
        extract_audio(input_file, audio_file)

        silence_percentage = estimate_silence(audio_file)
        if silence_percentage > 10:
            print(f"Estimated silence: {silence_percentage:.2f}%. Applying unsilence.")
            unsilenced_audio_file = os.path.join(temp_dir, "temp_unsilenced_audio.wav")
            subprocess.run(["unsilence", audio_file, unsilenced_audio_file, 
                            "-as", str(audible_speed), "-ss", str(silent_speed)], check=True)

            # Replace audio in the video with the unsilenced audio
            subprocess.run(["ffmpeg", "-i", input_file, "-i", unsilenced_audio_file, 
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", 
                            "-b:a", "128k", output_file], check=True)
        else:
            print(f"Estimated silence: {silence_percentage:.2f}%. Skipping unsilence.")
            shutil.copy(input_file, output_file)

        shutil.rmtree(temp_dir)
    else:
        # Process as audio file
        silence_percentage = estimate_silence(input_file)
        if silence_percentage > 10:
            print(f"Estimated silence: {silence_percentage:.2f}%. Applying unsilence.")
            subprocess.run(["unsilence", input_file, output_file, 
                            "-as", str(audible_speed), "-ss", str(silent_speed)], check=True)
        else:
            print(f"Estimated silence: {silence_percentage:.2f}%. Skipping unsilence.")
            shutil.copy(input_file, output_file)

def normalize(input_file, output_file):
    """
    Normalize audio levels using ffmpeg-normalize for audio/video files.
    
    Args:
    input_file (str): Path to the input audio/video file.
    output_file (str): Path to save the normalized audio/video.
    
    Returns:
    None
    """
    if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Extract audio from video
        temp_dir = tempfile.mkdtemp()
        audio_file = os.path.join(temp_dir, "temp_audio.wav")
        extract_audio(input_file, audio_file)

        normalized_audio_file = os.path.join(temp_dir, "temp_normalized_audio.wav")
        subprocess.run([
            "ffmpeg-normalize", "-pr", "-nt", "rms", audio_file, 
            "-prf", "highpass=f=200", "-prf", "dynaudnorm=p=0.4:s=15", "-pof", "lowpass=f=7000", 
            "-ar", "48000", "-c:a", "pcm_s16le", "--keep-loudness-range-target", 
            "-o", normalized_audio_file
        ], check=True)

        # Replace audio in the video with the normalized audio
        subprocess.run(["ffmpeg", "-i", input_file, "-i", normalized_audio_file, 
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", 
                        "-b:a", "128k", output_file], check=True)

        shutil.rmtree(temp_dir)
    else:
        # Process as audio file
        subprocess.run([
            "ffmpeg-normalize", "-pr", "-nt", "rms", input_file, 
            "-prf", "highpass=f=200", "-prf", "dynaudnorm=p=0.4:s=15", "-pof", "lowpass=f=7000", 
            "-ar", "48000", "-c:a", "pcm_s16le", "--keep-loudness-range-target", 
            "-o", output_file
        ], check=True)

def pipeline(input_file, output_dir, verbose=False):
    """
    Execute the complete processing pipeline: normalization, silence removal,
    audio extraction, transcription, and audio analysis.
    
    Args:
    input_file (str): Path to the input video or audio file.
    output_dir (str): Directory to save all output files.
    verbose (bool): If True, print detailed progress information.
    
    Returns:
    None
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Normalize
    normalized_file = os.path.join(output_dir, f"{base_name}_normalized.wav")
    normalize(input_file, normalized_file)
    
    # Unsilence
    unsilenced_file = os.path.join(output_dir, f"{base_name}_unsilenced.wav")
    unsilence_audio(normalized_file, unsilenced_file)
    
    # Extract audio (if input is video)
    if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        audio_file = os.path.join(output_dir, f"{base_name}.wav")
        extract_audio(input_file, audio_file)
    else:
        audio_file = unsilenced_file
    
    # Transcribe
    transcription_file = os.path.join(output_dir, f"{base_name}_transcription.txt")
    run_whisper(audio_file, transcription_file, verbose)
    
    # Analyze audio
    run_sonic_annotator(audio_file, output_dir)

def move_files(source_dir, dest_dir):
    """
    Move all files from source directory to destination directory.
    
    Args:
    source_dir (str): Source directory containing files to move.
    dest_dir (str): Destination directory to move files to.
    
    Returns:
    None
    """
    for file in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, file), dest_dir)

def check_ram(file_size):
    """
    Determine whether to use /tmp or /var/tmp based on available RAM.
    
    Args:
    file_size (int): Size of the file to be processed in bytes.
    
    Returns:
    str: Path to the appropriate temporary directory.
    """
    available_ram = psutil.virtual_memory().available
    return "/tmp" if available_ram > file_size * 2 else "/var/tmp"

def main():
    """
    Main function to parse command-line arguments and execute the chosen action.
    """
    parser = argparse.ArgumentParser(description="Video and Audio Processing Script")
    parser.add_argument("input_file", help="Input video or audio file")
    parser.add_argument("-o", "--output_dir", help="Output directory", default=".")
    parser.add_argument("-a", "--action", choices=["transcode", "extract_audio", "normalize", 
                                                   "unsilence", "pipeline"], required=True,
                        help="Action to perform on the input file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    file_size = os.path.getsize(args.input_file)
    temp_dir = check_ram(file_size)
    
    with tempfile.TemporaryDirectory(dir=temp_dir) as temp_output_dir:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        
        if args.verbose:
            print(f"Processing {args.input_file}")
            print(f"Using temporary directory: {temp_dir}")
        
        if args.action == "transcode":
            output_file = os.path.join(temp_output_dir, f"{base_name}_transcoded.mp4")
            transcode(args.input_file, output_file)
        elif args.action == "extract_audio":
            output_file = os.path.join(temp_output_dir, f"{base_name}.wav")
            extract_audio(args.input_file, output_file)
        elif args.action == "normalize":
            output_file = os.path.join(temp_output_dir, f"{base_name}_normalized.wav")
            normalize(args.input_file, output_file)
        elif args.action == "unsilence":
            output_file = os.path.join(temp_output_dir, f"{base_name}_unsilenced.wav")
            unsilence_audio(args.input_file, output_file)
        elif args.action == "pipeline":
            pipeline(args.input_file, temp_output_dir, args.verbose)
        
        move_files(temp_output_dir, args.output_dir)
    
    if args.verbose:
        print(f"Processing complete. Output files are in {args.output_dir}")
    else:
        print("Processing complete.")

if __name__ == "__main__":
    main()


# The code you provided defines a Python script named process_videov2.py designed for audio and video processing tasks. It leverages various external tools like ffmpeg, whisper.cpp, sonic-annotator, unsilence, and ffmpeg-normalize to perform actions such as:

# Transcoding: Converting video files to MP4 format with specific encoding settings.
# Audio Extraction: Extracting audio tracks from video files and saving them as WAV files.
# Normalization: Adjusting audio levels to a consistent target.
# Silence Removal: Detecting and optionally removing silent segments from audio or video.
# Transcription: Generating text transcripts from audio using the Whisper speech recognition model.
# Audio Analysis: Analyzing audio features like tempo and key using Sonic Annotator.
# The script is structured into several functions, each responsible for a specific task:

# run_whisper: Executes the Whisper model for audio transcription, handling potential errors and retries.
# run_sonic_annotator: Runs Sonic Annotator to analyze audio features and save the results.
# transcode: Uses ffmpeg to transcode video files to MP4 format.
# extract_audio: Extracts audio from video files using ffmpeg.
# estimate_silence: Calculates the percentage of silence in an audio file using the unsilence library and pydub.
# unsilence_audio: Removes silence from audio or video files if the silence percentage exceeds a threshold, utilizing the unsilence command.
# normalize: Normalizes audio levels in audio or video files using ffmpeg-normalize.
# pipeline: Executes a sequence of processing steps: normalization, silence removal, audio extraction (if applicable), transcription, and audio analysis.
# move_files: Moves files from a source directory to a destination directory.
# check_ram: Determines the appropriate temporary directory (/tmp or /var/tmp) based on available RAM and file size.
# main: Parses command-line arguments, determines the appropriate temporary directory, and calls the requested processing function.
# The script can be invoked from the command line with various options to specify the input file, output directory, desired action (transcode, extract_audio, normalize, unsilence, or pipeline), and verbosity level.

# In essence, this script provides a convenient way to automate common audio and video processing tasks, making it a useful tool for tasks like media preparation, analysis, and transcription