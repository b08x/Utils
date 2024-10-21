#!/usr/bin/env python 

import os
import sys
import argparse
from dotenv import load_dotenv
import videogrep  # Make sure videogrep is installed
import progressbar  # Make sure progressbar2 is installed

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


load_dotenv()


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

    project_path = utils.create_project_folder(args.input, args.output)
    print(f"Created project folder: {project_path}")

    try:
        if args.input.endswith(".json"):
            transcript = load_transcript(args.input)
            results = utils.process_transcript(transcript, project_path, args.topics)
            print(
                "Note: Video splitting and Gemini analysis are not performed when processing a transcript file."
            )
        else:
            results = utils.process_video(
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
