import os
import sys
import json
import progressbar
import google.generativeai as genai
from PIL import Image
from moviepy.editor import VideoFileClip

parent_directory = os.path.abspath('..')

sys.path.append(parent_directory)


from vts import (
    audio_processing,
    metadata_generation,
    segment_analysis,
    topic_modeling,
    utils,
)

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