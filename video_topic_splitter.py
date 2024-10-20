import os
import argparse
import json
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq
import sys
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence
import spacy
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
import videogrep
from moviepy.editor import VideoFileClip
import progressbar

load_dotenv()

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
API_CALL_DELAY = 1  # Delay between API calls in seconds

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def create_project_folder(input_video, base_output_dir):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    project_name = f"{base_name}_{timestamp}"
    project_path = os.path.join(base_output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)
    return project_path


def extract_audio(video_path, output_path):
    print("Extracting audio from video...")
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(output_path, format="wav")
        print("Audio extraction complete.")
    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        raise


def preprocess_audio(input_file, output_file):
    print("Preprocessing audio...")
    try:
        audio = AudioSegment.from_wav(input_file)
        high_passed = audio.high_pass_filter(60)
        normalized = high_passed.normalize()
        normalized.export(output_file, format="wav")
        print("Audio preprocessing complete.")
    except Exception as e:
        print(f"Error during audio preprocessing: {str(e)}")
        raise


def transcribe_file_deepgram(client, file_path, options, max_retries=3, retry_delay=5):
    print("Transcribing audio using Deepgram...")
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio:
                buffer_data = audio.read()
                payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/wav"}
                response = client.listen.rest.v("1").transcribe_file(payload, options)
            print("Transcription complete.")
            return response
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


def transcribe_file_groq(client, file_path, model="whisper-large-v3", language="en"):
    print("Transcribing audio using Groq...")
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model=model,
                language=language,
                response_format="text",
            )
        print("Transcription complete.")
        return transcription
    except Exception as e:
        print(f"Error during Groq transcription: {str(e)}")
        raise


def preprocess_text(text):
    print("Preprocessing text...")
    doc = nlp(text)
    subjects = []

    for sent in doc.sents:
        for token in sent:
            if "subj" in token.dep_:
                # Found a subject
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    subject = get_compound_subject(token)
                    subjects.append(subject)

    # Lemmatize and clean the subjects
    cleaned_subjects = [
        [
            token.lemma_.lower()
            for token in nlp(subject)
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        for subject in subjects
    ]

    # Remove duplicates and empty lists
    cleaned_subjects = [
        list(s) for s in set(tuple(sub) for sub in cleaned_subjects) if s
    ]

    print(
        f"Text preprocessing complete. Extracted {len(cleaned_subjects)} unique subjects."
    )
    return cleaned_subjects


def get_compound_subject(token):
    """
    Helper function to get the full compound subject if it exists.
    """
    subject = [token.text]

    # Check for compound subjects to the left
    for left_token in token.lefts:
        if left_token.dep_ == "compound":
            subject.insert(0, left_token.text)

    # Check for compound subjects to the right
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
            continue  # Skip sentences with no identifiable subjects

        # Use all subjects for topic identification
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


def split_video(input_video, segments, output_dir):
    print("Splitting video into segments...")
    try:
        video = VideoFileClip(input_video)
        for i, segment in enumerate(progressbar.progressbar(segments)):
            start_time = segment["start"]
            end_time = segment["end"]
            segment_clip = video.subclip(start_time, end_time)
            output_path = os.path.join(output_dir, f"segment_{i+1}.mp4")
            segment_clip.write_videofile(
                output_path, codec="libx264", audio_codec="aac"
            )
        video.close()
        print("Video splitting complete.")
    except Exception as e:
        print(f"Error during video splitting: {str(e)}")
        raise


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


def save_results(results, project_path):
    print("Saving results...")
    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


def process_video(video_path, project_path, api="deepgram", num_topics=5):
    print(f"Processing video: {video_path}")
    audio_dir = os.path.join(project_path, "audio")
    transcript_dir = os.path.join(project_path, "transcript")
    segments_dir = os.path.join(project_path, "segments")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)

    try:
        raw_audio_path = os.path.join(audio_dir, "extracted_audio.wav")
        extract_audio(video_path, raw_audio_path)

        preprocessed_audio_path = os.path.join(audio_dir, "preprocessed_audio.wav")
        preprocess_audio(raw_audio_path, preprocessed_audio_path)

        print("Parsing transcript with Videogrep...")
        transcript = videogrep.parse_transcript(video_path)
        print("Transcript parsing complete.")

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

        if not transcript:
            print("No transcript found. Transcribing audio...")
            if api == "deepgram":
                response = transcribe_file_deepgram(
                    deepgram_client, preprocessed_audio_path, deepgram_options
                )
                transcription = json.loads(response.to_json())
                transcript = [
                    {
                        "content": utterance["transcript"],
                        "start": utterance["start"],
                        "end": utterance["end"],
                    }
                    for utterance in transcription["results"]["utterances"]
                ]
            else:  # Groq
                transcription = transcribe_file_groq(
                    groq_client, preprocessed_audio_path
                )
                transcript = [
                    {
                        "content": transcription,
                        "start": 0,
                        "end": AudioSegment.from_wav(
                            preprocessed_audio_path
                        ).duration_seconds,
                    }
                ]

        full_text = " ".join([sentence["content"] for sentence in transcript])
        preprocessed_subjects = preprocess_text(full_text)
        lda_model, corpus, dictionary = perform_topic_modeling(
            preprocessed_subjects, num_topics
        )

        segments = identify_segments(transcript, lda_model, dictionary, num_topics)

        metadata = generate_metadata(segments, lda_model)

        results = {
            "transcription": full_text,
            "api_used": api,
            "topics": [
                {
                    "topic_id": topic_id,
                    "words": [
                        word for word, _ in lda_model.show_topic(topic_id, topn=10)
                    ],
                }
                for topic_id in range(num_topics)
            ],
            "segments": metadata,
        }

        # Save results before video splitting
        save_results(results, project_path)

        # Split video after saving results
        # split_video(video_path, segments, segments_dir)

        print("Cleaning up temporary files...")
        os.remove(raw_audio_path)
        os.remove(preprocessed_audio_path)

        print("Processing complete.")
        return results
    except Exception as e:
        print(f"An error occurred during video processing: {str(e)}")
        # Save partial results if available
        if "results" in locals():
            save_results(results, project_path)
            print("Partial results have been saved.")
        raise


def main():
    parser = argparse.ArgumentParser(description="Split video based on topics")
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input video file"
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
    args = parser.parse_args()

    project_path = create_project_folder(args.input, args.output)
    print(f"Created project folder: {project_path}")

    try:
        results = process_video(args.input, project_path, args.api, args.topics)

        print(f"\nProcessing complete. Project folder: {project_path}")
        print(f"Results saved in: {os.path.join(project_path, 'results.json')}")
        print("\nTop words for each topic:")
        for topic in results["topics"]:
            print(f"Topic {topic['topic_id'] + 1}: {', '.join(topic['words'])}")
        print(f"\nGenerated {len(results['segments'])} video segments")
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        print("Please check the project folder for any partial results or logs.")


if __name__ == "__main__":
    main()
