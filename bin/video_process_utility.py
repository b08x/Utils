#!/usr/bin/env python

import os
import argparse
import json
import sys
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource, DeepgramError
from groq import Groq
import time
from pydub import AudioSegment
import spacy
from gensim import corpora
from gensim.models import LdaMulticore
import videogrep
from moviepy.editor import VideoFileClip
import google.generativeai as genai
from PIL import Image
import traceback

load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_audio(video_path, output_path=None):
    try:
        audio = AudioSegment.from_file(video_path)
        if output_path:
            audio.export(output_path, format="wav")
            return f"Audio extracted to {output_path}"
        else:
            return audio.raw_data
    except Exception as e:
        return f"Error during audio extraction: {str(e)}"

def normalize_audio(input_file, output_file, lowpass_freq=1000, highpass_freq=100, norm_level=-3,
                    compression_params=None):
    """
    Normalize audio using PySox with optional low-pass and high-pass filtering and compression.
    """
    try:
        tfm = sox.Transformer()
        
        # Apply low-pass filter
        if lowpass_freq:
            tfm.lowpass(frequency=lowpass_freq)
        
        # Apply high-pass filter
        if highpass_freq:
            tfm.highpass(frequency=highpass_freq)
        
        # Normalize audio
        tfm.norm(db_level=norm_level)
        
        # Apply compression if parameters are provided
        if compression_params:
            tfm.compand(**compression_params)
        
        # Build and apply the effects
        tfm.build(input_file, output_file)
        
        return {
            "status": "success",
            "message": f"Audio normalized and saved to {output_file}",
            "lowpass_freq": lowpass_freq,
            "highpass_freq": highpass_freq,
            "norm_level": norm_level,
            "compression_applied": bool(compression_params)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during audio normalization: {str(e)}"
        }

def transcribe_file_deepgram(file_path, options=None):
    try:
        deepgram_key = os.getenv("DG_API_KEY")
        if not deepgram_key:
            return "DG_API_KEY environment variable is not set"
        
        client = DeepgramClient(deepgram_key)
        if options is None:
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                language="en",
                punctuate=True,
                utterances=True,
                diarize=True,
            )
        
        with open(file_path, "rb") as audio:
            buffer_data = audio.read()
            payload = {"buffer": buffer_data, "mimetype": "audio/wav"}
            response = client.listen.rest.v("1").transcribe_file(payload, options)
        return json.loads(response.to_json())
    except Exception as e:
        return f"Error during Deepgram transcription: {str(e)}"

def transcribe_file_groq(file_path, model="whisper-large-v3", language="en", prompt=None):
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            return "GROQ_API_KEY environment variable is not set"
        
        client = Groq(api_key=groq_key)
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model=model,
                language=language,
                response_format="json",
                prompt=prompt
            )
        return json.loads(transcription.text)
    except Exception as e:
        return f"Error during Groq transcription: {str(e)}"

def preprocess_text(text):
    doc = nlp(text)
    subjects = []
    for sent in doc.sents:
        for token in sent:
            if "subj" in token.dep_:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    subject = get_compound_subject(token)
                    subjects.append(subject)
    
    cleaned_subjects = [
        [token.lemma_.lower() for token in nlp(subject) 
         if not token.is_stop and not token.is_punct and token.is_alpha]
        for subject in subjects
    ]
    cleaned_subjects = [list(s) for s in set(tuple(sub) for sub in cleaned_subjects) if s]
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
    dictionary = corpora.Dictionary(subjects)
    corpus = [dictionary.doc2bow(subject) for subject in subjects]
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                             chunksize=100, passes=10, per_word_topics=True)
    topics = [{"topic_id": topic_id, "words": [word for word, _ in lda_model.show_topic(topic_id, topn=10)]} 
              for topic_id in range(num_topics)]
    return topics

def identify_segments(transcript, lda_model, dictionary, num_topics):
    segments = []
    current_segment = {"start": 0, "end": 0, "content": "", "topic": None}
    
    for sentence in transcript:
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
                "topic": dominant_topic
            }
        else:
            current_segment["end"] = sentence["end"]
            current_segment["content"] += " " + sentence["content"]
    
    if current_segment["content"]:
        segments.append(current_segment)
    
    return segments

def analyze_segment_with_gemini(segment_path, transcript):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-pro-vision')

        video = VideoFileClip(segment_path)
        frame = video.get_frame(0)
        image = Image.fromarray(frame)
        video.close()

        prompt = f"Analyze this video segment. The transcript for this segment is: '{transcript}'. Describe the main subject matter, key visual elements, and how they relate to the transcript."

        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error during Gemini analysis: {str(e)}"

def split_video(input_video, segments, output_dir):
    try:
        video = VideoFileClip(input_video)
        split_info = []
        for i, segment in enumerate(segments):
            start_time = segment["start"]
            end_time = segment["end"]
            segment_clip = video.subclip(start_time, end_time)
            output_path = os.path.join(output_dir, f"segment_{i+1}.mp4")
            segment_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            split_info.append({
                "segment_id": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "output_path": output_path
            })
        video.close()
        return split_info
    except Exception as e:
        return f"Error during video splitting: {str(e)}"

def full_pipeline(input_video, output_dir, api="deepgram", num_topics=5, groq_prompt=None, 
                  lowpass_freq=1000, highpass_freq=100, norm_level=-3, compress=False):
    """
    Execute the full video processing pipeline.
    """
    try:
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        audio_dir = os.path.join(output_dir, "audio")
        segments_dir = os.path.join(output_dir, "segments")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(segments_dir, exist_ok=True)

        # Extract audio
        raw_audio_path = os.path.join(audio_dir, "extracted_audio.wav")
        extract_audio(input_video, raw_audio_path)

        # Normalize audio
        normalized_audio_path = os.path.join(audio_dir, "normalized_audio.wav")
        compression_params = {
            "attack_time": 0.2,
            "decay_time": 1,
            "soft_knee_db": 2,
            "threshold": -20,
            "db_from": -20,
            "db_to": -10
        } if compress else None
        normalize_audio(raw_audio_path, normalized_audio_path, lowpass_freq, highpass_freq, norm_level, compression_params)

        # Transcribe audio
        if api == "deepgram":
            transcription = transcribe_file_deepgram(normalized_audio_path)
            transcript = [
                {
                    "content": utterance["transcript"],
                    "start": utterance["start"],
                    "end": utterance["end"]
                }
                for utterance in transcription.get('results', {}).get('utterances', [])
            ]
        else:  # Groq
            transcription = transcribe_file_groq(normalized_audio_path, prompt=groq_prompt)
            if isinstance(transcription, dict) and 'segments' in transcription:
                transcript = [
                    {
                        "content": segment.get('text', ''),
                        "start": segment.get('start', 0),
                        "end": segment.get('end', 0)
                    }
                    for segment in transcription['segments']
                ]
            elif isinstance(transcription, str):
                # If Groq returns a string, create a single segment
                transcript = [{
                    "content": transcription,
                    "start": 0,
                    "end": 0  # You might want to get the audio duration here
                }]
            else:
                raise ValueError(f"Unexpected transcription format from Groq: {type(transcription)}")

        # Save transcript
        transcript_path = os.path.join(output_dir, "transcript.json")
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2)

        # Preprocess text and perform topic modeling
        full_text = " ".join([sentence["content"] for sentence in transcript])
        preprocessed_subjects = preprocess_text(full_text)
        topics = perform_topic_modeling(preprocessed_subjects, num_topics)

        # Identify segments
        segments = identify_segments(transcript, topics, preprocessed_subjects, num_topics)

        # Split video and analyze segments
        split_info = split_video(input_video, segments, segments_dir)

        # Analyze segments with Gemini
        analyzed_segments = []
        for segment in split_info:
            gemini_analysis = analyze_segment_with_gemini(segment['output_path'], segment['transcript'])
            analyzed_segments.append({
                **segment,
                "gemini_analysis": gemini_analysis
            })

        # Prepare final results
        results = {
            "input_video": input_video,
            "normalized_audio": normalized_audio_path,
            "transcription_api": api,
            "transcript_path": transcript_path,
            "topics": topics,
            "segments": analyzed_segments
        }

        # Save results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return {
            "status": "success",
            "message": "Full pipeline completed successfully",
            "results_path": results_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during full pipeline execution: {str(e)}",
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description="Video processing utility functions")
    parser.add_argument("function", nargs='?', choices=["extract_audio", "transcribe_deepgram", "transcribe_groq", 
                                             "preprocess_text", "topic_modeling", "identify_segments", 
                                             "analyze_segment", "split_video", "normalize_audio"],
                        help="Function to execute (if not specified, full pipeline will be executed)")
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", help="Output file or directory path")
    parser.add_argument("--topics", type=int, default=5, help="Number of topics for LDA model")
    parser.add_argument("--api", choices=["deepgram", "groq"], default="deepgram", help="Transcription API")
    parser.add_argument("--prompt", help="Prompt for Groq transcription or Gemini analysis")
    parser.add_argument("--lowpass", type=int, default=1000, help="Low-pass filter frequency")
    parser.add_argument("--highpass", type=int, default=100, help="High-pass filter frequency")
    parser.add_argument("--norm-level", type=float, default=-3, help="Normalization level in dB")
    parser.add_argument("--compress", action="store_true", help="Apply compression")
    args = parser.parse_args()

    if args.function is None:
        # Execute full pipeline
        result = full_pipeline(args.input, args.output or os.getcwd(), args.api, args.topics, args.prompt,
                               args.lowpass, args.highpass, args.norm_level, args.compress)
    else:
        # Execute specific function
        result = None
        if args.function == "normalize_audio":
            compression_params = {
                "attack_time": 0.2,
                "decay_time": 1,
                "soft_knee_db": 2,
                "threshold": -20,
                "db_from": -20,
                "db_to": -10
            } if args.compress else None
            
            result = normalize_audio(args.input, args.output, args.lowpass, args.highpass, args.norm_level, compression_params)
        elif args.function == "extract_audio":
            result = extract_audio(args.input, args.output)
        elif args.function == "transcribe_deepgram":
            result = transcribe_file_deepgram(args.input)
        elif args.function == "transcribe_groq":
            result = transcribe_file_groq(args.input, prompt=args.prompt)
        elif args.function == "preprocess_text":
            with open(args.input, 'r') as f:
                text = f.read()
            result = preprocess_text(text)
        elif args.function == "topic_modeling":
            with open(args.input, 'r') as f:
                subjects = json.load(f)
            result = perform_topic_modeling(subjects, args.topics)
        elif args.function == "identify_segments":
            # This requires more complex input, you might need to adjust based on your needs
            pass
        elif args.function == "analyze_segment":
            with open(args.input, 'r') as f:
                transcript = f.read()
            result = analyze_segment_with_gemini(args.output, transcript)
        elif args.function == "split_video":
            with open(args.input, 'r') as f:
                segments = json.load(f)
            result = split_video(args.output, segments, os.path.dirname(args.output))

    json.dump(result, sys.stdout, indent=2)

if __name__ == "__main__":
    main()