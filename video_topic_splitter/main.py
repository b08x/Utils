#!/usr/bin/env python3
"""Command-line interface for video topic splitter."""

import os
import argparse
import sys
parent_directory = os.path.abspath('..')

sys.path.append(parent_directory)

# from core import process_video
# from project import create_project_folder, load_checkpoint
# from constants import CHECKPOINTS
# from transcription import load_transcript
# from core import process_transcript

from video_topic_splitter import (
    core,
    project,
    constants,
    transcription,
    video_analysis
)

def main():
    """Main entry point for the CLI."""
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
        "--topics",
        type=int,
        default=5,
        help="Number of topics for LDA model"
    )
    parser.add_argument(
        "--groq-prompt",
        help="Optional prompt for Groq transcription"
    )
    args = parser.parse_args()

    project_path = project.create_project_folder(args.input, args.output)
    print(f"Project folder: {project_path}")

    try:
        checkpoint = project.load_checkpoint(project_path)
        if checkpoint and checkpoint['stage'] == constants.CHECKPOINTS['PROCESS_COMPLETE']:
            results = checkpoint['data']['results']
            print("Loading results from previous complete run.")
        elif args.input.endswith(".json"):
            transcript = transcription.load_transcript(args.input)
            results = core.process_transcript(transcript, project_path, args.topics)
            print(
                "Note: Video splitting and Gemini analysis are not performed when processing a transcript file."
            )
        else:
            results = core.process_video(
                args.input,
                project_path,
                args.api,
                args.topics,
                args.groq_prompt
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
        print("You can resume processing by running the script again with the same arguments.")

if __name__ == "__main__":
    main()