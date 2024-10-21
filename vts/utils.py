import os
import sys
import time
import json
import progressbar
import videogrep  # Make sure videogrep is installed
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq

parent_directory = os.path.abspath('..')

sys.path.append(parent_directory)


from vts import (
    audio_processing,
    metadata_generation,
    segment_analysis,
    topic_modeling,
    utils,
)


def create_project_folder(input_path, base_output_dir):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    project_name = f"{base_name}_{timestamp}"
    project_path = os.path.join(base_output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path

def transcribe_file_deepgram(client, file_path, options, max_retries=3, retry_delay=5):
    print("Transcribing audio using Deepgram...")
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/mp4"}
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            print("Transcription complete.")
            return json.loads(response.to_json())
        except DeepgramError as e:
            if attempt < max_retries - 1:
                print(
                    f"API call failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                print(f"Transcription failed after {max_retries} attempts: {str(e)}")
                raise
        except Exception as e:
            print(f"Unexpected error during transcription: {str(e)}")
            raise


def transcribe_file_groq(
    client, file_path, model="whisper-large-v3", language="en", prompt=None
):
    print("Transcribing audio using Groq...")
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model=model,
                prompt=prompt,
                response_format="verbose_json",
                language=language,
                temperature=0.2
            )
        print("Transcription complete.")
        return json.loads(transcription.text)
    except Exception as e:
        print(f"Error during Groq transcription: {str(e)}")
        raise

def handle_audio_video(video_path, project_path):
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    normalized_video_path = os.path.join(project_path, "normalized_video.mkv")
    normalize_result = audio_processing.normalize_audio(video_path, normalized_video_path)
    if normalize_result["status"] == "error":
        print(f"Error during audio normalization: {normalize_result['message']}")
        # Handle the error (e.g., exit or continue without normalization)
    else:
        print(normalize_result["message"])

    # Remove silence
    unsilenced_video_path = os.path.join(project_path, "unsilenced_video.mkv")
    silence_removal_result = audio_processing.remove_silence(
        normalized_video_path, unsilenced_video_path
    )
    if silence_removal_result["status"] == "error":
        print(f"Error during silence removal: {silence_removal_result['message']}")
        # Handle the error (e.g., exit or continue without silence removal)
    else:
        print(silence_removal_result["message"])

    # Extract audio from unsilenced video
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.wav")
    audio_processing.extract_audio(unsilenced_video_path, raw_audio_path)

    # Convert to mono and resample for transcription
    mono_resampled_audio_path = os.path.join(audio_dir, "mono_resampled_audio.m4a")
    conversion_result = audio_processing.convert_to_mono_and_resample(
        raw_audio_path, mono_resampled_audio_path
    )
    if conversion_result["status"] == "error":
        print(f"Error during audio conversion: {conversion_result['message']}")
        # Handle the error (e.g., exit or continue without conversion)
    else:
        print(conversion_result["message"])

    return unsilenced_video_path, mono_resampled_audio_path


def handle_transcription(
    video_path, audio_path, project_path, api="deepgram", num_topics=2, groq_prompt=None
):
    segments_dir = os.path.join(project_path, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    print("Parsing transcript with Videogrep...")
    transcript = videogrep.parse_transcript(video_path)
    print("Transcript parsing complete.")

    if not transcript:
        print("No transcript found. Transcribing audio...")
        deepgram_key = os.getenv("DG_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not deepgram_key:
            raise ValueError("DG_API_KEY environment variable is not set")
        if not groq_key and api == "groq":
            raise ValueError("GROQ_API_KEY environment variable is not set")

        if api == "deepgram":
            deepgram_client = DeepgramClient(deepgram_key)
            deepgram_options = PrerecordedOptions(
                model="nova-2",
                language="en",
                topics=True,
                intents=True,
                smart_format=True,
                punctuate=True,
                paragraphs=True,
                utterances=True,
                diarize=True,
                filler_words=True,
                sentiment=True,
            )
            # Transcribe the normalized audio
            transcription = transcribe_file_deepgram(
                deepgram_client, audio_path, deepgram_options
            )
            transcript = [
                {
                    "content": utterance["transcript"],
                    "start": utterance["start"],
                    "end": utterance["end"],
                }
                for utterance in transcription["results"]["utterances"]
            ]
        else:  # Groq
            groq_client = Groq(api_key=groq_key)
            # Transcribe the normalized audio
            transcription = transcribe_file_groq(
                groq_client, audio_path, prompt=groq_prompt
            )
            transcript = [
                {
                    "content": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                }
                for segment in transcription["segments"]
            ]

    metadata_generation.save_transcription(transcription, project_path)

    metadata_generation.save_transcript(transcript, project_path)
    results = process_transcript(transcript, project_path, num_topics)

    # Split the video and analyze segments
    analyzed_segments = segment_analysis.split_and_analyze_video(
        video_path, results["segments"], segments_dir
    )

    # Update results with analyzed segments
    results["analyzed_segments"] = analyzed_segments

    # Save updated results
    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # print("Cleaning up temporary files...")
    # os.remove(raw_audio_path)
    # os.remove(normalized_audio_path) # Uncomment to remove normalized audio

    return results


def process_video(
    video_path, project_path, api="deepgram", num_topics=2, groq_prompt=None
):

    unsilenced_video_path, mono_resampled_audio_path = handle_audio_video(
        video_path, project_path
    )
    results = handle_transcription(
        unsilenced_video_path,
        mono_resampled_audio_path,
        project_path,
        api,
        num_topics,
        groq_prompt,
    )

    return results


def process_transcript(transcript, project_path, num_topics=5):
    full_text = " ".join([sentence["content"] for sentence in transcript])
    preprocessed_subjects = topic_modeling.preprocess_text(full_text)
    lda_model, corpus, dictionary = topic_modeling.perform_topic_modeling(
        preprocessed_subjects, num_topics
    )

    segments = topic_modeling.identify_segments(transcript, lda_model, dictionary, num_topics)
    metadata = metadata_generation.generate_metadata(segments, lda_model)

    results = {
        "topics": [
            {
                "topic_id": topic_id,
                "words": [word for word, _ in lda_model.show_topic(topic_id, topn=10)],
            }
            for topic_id in range(num_topics)
        ],
        "segments": metadata,
    }

    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results
