"""
Video Topic Splitter - A tool for segmenting videos based on topic analysis.

This package was generated by Claude (Anthropic) based on detailed specifications 
provided by the user. 

Technical Specifications (user-provided):
Audio Processing:
- FFmpeg parameters: highpass=f=200, acompressor threshold=-12dB:ratio=4:attack=5:release=50
- Sample rate: 16000Hz for speech recognition
- Normalization: RMS-based with -9.0 dB target
- Silence parameters: 1.5s duration, -25dB threshold

Models & Parameters:
- Deepgram: nova-2 model with topics, intents, smart_format, paragraphs, diarize settings
- Groq: whisper-large-v3 model with temperature=0.2
- Gemini: gemini-1.5-pro-latest for visual analysis
- Topic Modeling: LDA with num_topics=5, passes=10, chunksize=100, random_state=100

Prompt Engineering:
- Gemini analysis prompt template for video segments
- Groq transcription contextual prompting

Implementation Details (Claude-generated):
- Code structure and organization
- Error handling and recovery mechanisms
- Package architecture
- Documentation

For more information, see the README.md file.
"""

__version__ = "0.1.0"
__author__ = "Generated by Claude (Anthropic)"
__credits__ = ["User: Original concept and technical specifications",
               "Claude (Anthropic): Code implementation"]


# Import main functionality to make it available at package level
from video_topic_splitter.core import process_video
from video_topic_splitter.project import create_project_folder, save_checkpoint, load_checkpoint