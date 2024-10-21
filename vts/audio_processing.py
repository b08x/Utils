import os
import sys
import subprocess
import json
import progressbar
from pydub import AudioSegment

parent_directory = os.path.abspath('..')

sys.path.append(parent_directory)


from vts import (
    audio_processing,
    metadata_generation,
    segment_analysis,
    topic_modeling,
    utils,
)

def convert_to_mono_and_resample(input_file, output_file, sample_rate=16000):
    """Converts audio to mono, resamples, applies gain control, and a high-pass filter."""
    try:
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            "highpass=f=200, acompressor=threshold=-12dB:ratio=4:attack=5:release=50, loudnorm",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",  # Use AAC codec for M4A
            output_file,
        ]
        subprocess.run(command, check=True)
        return {
            "status": "success",
            "message": f"Audio converted to mono, resampled to {sample_rate}Hz, gain-adjusted, high-pass filtered, and saved to {output_file}",
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Error during audio conversion: {str(e)}",
        }


def normalize_audio(input_file, output_file, lowpass_freq=8000, highpass_freq=100):
    """Normalizes audio using ffmpeg-normalize."""
    try:
        command = [
            "ffmpeg-normalize",
            "-pr",  # Preserve ReplayGain tags
            "-tp",
            "-3.0",
            "-nt",
            "rms",
            input_file,
            "-prf",
            f"highpass=f={highpass_freq}",
            "-prf",
            "dynaudnorm=p=0.4:s=15, loudnorm",
            "-pof",
            f"lowpass=f={lowpass_freq}, loudnorm",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            "--keep-loudness-range-target",
            "-o",
            output_file,
        ]
        subprocess.run(command, check=True)
        return {
            "status": "success",
            "message": f"Audio normalized and saved to {output_file}",
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Error during audio normalization: {str(e)}",
        }


def remove_silence(input_file, output_file, duration="1.5", threshold="-25"):
    """Removes silence from audio using unsilence."""
    try:
        command = [
            "unsilence",
            "-y",  # non-interactive mode
            "-d",  # Delete silent parts
            "-ss",
            duration,  # Minimum silence duration
            "-sl",
            threshold,  # Silence threshold
            input_file,
            output_file,
        ]
        subprocess.run(command, check=True)
        return {
            "status": "success",
            "message": f"Silence removed from audio and saved to {output_file}",
        }
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Error during silence removal: {str(e)}"}


def extract_audio(video_path, output_path):
    print("Extracting audio from video...")
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(output_path, format="wav")
        print("Audio extraction complete.")
    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        raise