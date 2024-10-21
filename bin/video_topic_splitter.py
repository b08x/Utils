#!/usr/bin/env python

import os
import subprocess
import argparse
import json
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq
import sys
import time
from pydub import AudioSegment
import spacy
from gensim import corpora
from gensim.models import LdaMulticore
import videogrep
from moviepy.editor import VideoFileClip
import progressbar
import google.generativeai as genai
from PIL import Image

load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def create_project_folder(input_path, base_output_dir):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    project_name = f"{base_name}_{timestamp}"
    project_path = os.path.join(base_output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path


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
            "dynaudnorm=p=0.4:s=15",
            "-pof",
            f"lowpass=f={lowpass_freq}",
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


def preprocess_text(text):
    print("Preprocessing text...")
    doc = nlp(text)
    subjects = []

    for sent in doc.sents:
        for token in sent:
            if "subj" in token.dep_:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    subject = get_compound_subject(token)
                    subjects.append(subject)

    cleaned_subjects = [
        [
            token.lemma_.lower()
            for token in nlp(subject)
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        for subject in subjects
    ]

    cleaned_subjects = [
        list(s) for s in set(tuple(sub) for sub in cleaned_subjects) if s
    ]

    print(
        f"Text preprocessing complete. Extracted {len(cleaned_subjects)} unique subjects."
    )
    return cleaned_subjects


def get_compound_subject(token):
    subject = [token.text]
    for left_token in token.lefts:
        if left_token.dep_ == "compound":
            subject.insert(0, left_token.text)
    for right_token in token.rights:
        if right_token.dep_ == "compound":
            subject.append(right_token.text)
    return " ".join(subject)


def perform_topic_modeling(subjects, num_topics=5):
    print(f"Performing topic modeling with {num_topics} topics...")
    dictionary = corpora.Dictionary(subjects)
    corpus = [dictionary.doc2bow(subject) for subject in subjects]
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        chunksize=100,
        passes=10,
        per_word_topics=True,
    )
    print("Topic modeling complete.")
    return lda_model, corpus, dictionary


def identify_segments(transcript, lda_model, dictionary, num_topics):
    print("Identifying segments based on topics...")
    segments = []
    current_segment = {"start": 0, "end": 0, "content": "", "topic": None}

    for sentence in progressbar.progressbar(transcript):
        subjects = preprocess_text(sentence["content"])
        if not subjects:
            continue

        bow = dictionary.doc2bow([token for subject in subjects for token in subject])
        topic_dist = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0] if topic_dist else None

        if dominant_topic != current_segment["topic"]:
            if current_segment["content"]:
                current_segment["end"] = sentence["start"]
                segments.append(current_segment)
            current_segment = {
                "start": sentence["start"],
                "end": sentence["end"],
                "content": sentence["content"],
                "topic": dominant_topic,
            }
        else:
            current_segment["end"] = sentence["end"]
            current_segment["content"] += " " + sentence["content"]

    if current_segment["content"]:
        segments.append(current_segment)

    print(f"Identified {len(segments)} segments.")
    return segments


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


def process_transcript(transcript, project_path, num_topics=5):
    full_text = " ".join([sentence["content"] for sentence in transcript])
    preprocessed_subjects = preprocess_text(full_text)
    lda_model, corpus, dictionary = perform_topic_modeling(
        preprocessed_subjects, num_topics
    )

    segments = identify_segments(transcript, lda_model, dictionary, num_topics)
    metadata = generate_metadata(segments, lda_model)

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


def analyze_segment_with_gemini(segment_path, transcript):
    print(f"Analyzing segment: {segment_path}")
    # Set up the Gemini model
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Load the video segment as an image (first frame)
    video = VideoFileClip(segment_path)
    frame = video.get_frame(0)
    image = Image.fromarray(frame)
    video.close()

    # Prepare the prompt
    prompt = f"Analyze this video segment. The transcript for this segment is: '{transcript}'. Describe the main subject matter, key visual elements, and how they relate to the transcript."

    # Generate content
    response = model.generate_content([prompt, image])

    return response.text


def split_and_analyze_video(input_video, segments, output_dir):
    print("Splitting video into segments and analyzing...")
    try:
        video = VideoFileClip(input_video)
        analyzed_segments = []

        for i, segment in enumerate(progressbar.progressbar(segments)):
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            segment_clip = video.subclip(start_time, end_time)
            output_path = os.path.join(output_dir, f"segment_{i+1}.mp4")
            segment_clip.write_videofile(
                output_path, codec="libx264", audio_codec="aac"
            )

            # Analyze the segment with Gemini
            gemini_analysis = analyze_segment_with_gemini(
                output_path, segment["transcript"]
            )

            analyzed_segments.append(
                {
                    "segment_id": i + 1,
                    "start_time": start_time,
                    "end_time": end_time,
                    "transcript": segment["transcript"],
                    "topic": segment["dominant_topic"],
                    "keywords": segment["top_keywords"],
                    "gemini_analysis": gemini_analysis,
                }
            )

        video.close()
        print("Video splitting and analysis complete.")
        return analyzed_segments
    except Exception as e:
        print(f"Error during video splitting and analysis: {str(e)}")
        raise


def handle_audio_video(video_path, project_path):
    audio_dir = os.path.join(project_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    normalized_video_path = os.path.join(project_path, "normalized_video.mkv")
    normalize_result = normalize_audio(video_path, normalized_video_path)
    if normalize_result["status"] == "error":
        print(f"Error during audio normalization: {normalize_result['message']}")
        # Handle the error (e.g., exit or continue without normalization)
    else:
        print(normalize_result["message"])

    # Remove silence
    unsilenced_video_path = os.path.join(project_path, "unsilenced_video.mkv")
    silence_removal_result = remove_silence(
        normalized_video_path, unsilenced_video_path
    )
    if silence_removal_result["status"] == "error":
        print(f"Error during silence removal: {silence_removal_result['message']}")
        # Handle the error (e.g., exit or continue without silence removal)
    else:
        print(silence_removal_result["message"])

    # Extract audio from unsilenced video
    raw_audio_path = os.path.join(audio_dir, "extracted_audio.wav")
    extract_audio(unsilenced_video_path, raw_audio_path)

    # Convert to mono and resample for transcription
    mono_resampled_audio_path = os.path.join(audio_dir, "mono_resampled_audio.m4a")
    conversion_result = convert_to_mono_and_resample(
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

    save_transcription(transcription, project_path)

    save_transcript(transcript, project_path)
    results = process_transcript(transcript, project_path, num_topics)

    # Split the video and analyze segments
    analyzed_segments = split_and_analyze_video(
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
        unsilenced_video_path, mono_resampled_audio_path, project_path, api, num_topics, groq_prompt
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process video or transcript for topic-based segmentation and multi-modal analysis"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input video file or transcript JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.getcwd(),
        help="Base output directory for project folders",
    )
    parser.add_argument(
        "--api",
        choices=["deepgram", "groq"],
        default="deepgram",
        help="Choose API: deepgram or groq",
    )
    parser.add_argument(
        "--topics", type=int, default=5, help="Number of topics for LDA model"
    )
    parser.add_argument("--groq-prompt", help="Optional prompt for Groq transcription")
    args = parser.parse_args()

    project_path = create_project_folder(args.input, args.output)
    print(f"Created project folder: {project_path}")

    try:
        if args.input.endswith(".json"):
            transcript = load_transcript(args.input)
            results = process_transcript(transcript, project_path, args.topics)
            print(
                "Note: Video splitting and Gemini analysis are not performed when processing a transcript file."
            )
        else:
            results = process_video(
                args.input, project_path, args.api, args.topics, args.groq_prompt
            )

        print(f"\nProcessing complete. Project folder: {project_path}")
        print(f"Results saved in: {os.path.join(project_path, 'results.json')}")
        print("\nTop words for each topic:")
        for topic in results["topics"]:
            print(f"Topic {topic['topic_id'] + 1}: {', '.join(topic['words'])}")
        print(f"\nGenerated and analyzed {len(results['analyzed_segments'])} segments")

        if not args.input.endswith(".json"):
            print(f"Video segments saved in: {os.path.join(project_path, 'segments')}")
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        print("Please check the project folder for any partial results or logs.")


if __name__ == "__main__":
    main()
