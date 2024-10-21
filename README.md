# Utility Scripts

- [ ] index & descriptions


```shell
├── bin
│   ├── backup.sh
│   ├── brightness.rb
│   ├── brightness.sh
│   ├── cleanup.sh
│   ├── convert2ogg.sh
│   ├── disconnect_pulse.sh
│   ├── docker-backup.sh
│   ├── dots.sh
│   ├── guake-utils
│   ├── jack-load.sh
│   ├── mic_mute.sh
│   ├── process_video.sh
│   ├── remote_docker_compose_manager.sh
│   ├── ripfzf-vscode.sh
│   ├── search_web.sh
│   ├── set-govna.sh
│   ├── timer.sh
│   ├── whisper-stream
│   └── yt-dlp-transcript.sh
```


* anything in dev/staging are WIP scripts

* anything in defunct are scripts no longer in use


This Python script is designed to analyze video content by identifying key topics and splitting the video into segments based on those topics. It then uses a combination of NLP techniques, topic modeling, and potentially AI-powered analysis (like Google's Gemini) to provide a deeper understanding of the video's content.

Here's a breakdown of the script's functionality:

1. Preprocessing:

Audio Extraction and Enhancement: If the input is a video file, the script first extracts the audio. It then applies audio normalization, silence removal, and format conversion to prepare it for transcription.
Transcription: The script can transcribe the audio using either the Deepgram or Groq API. You can choose the API using the --api argument.
Text Preprocessing: The transcript is processed using spaCy, a natural language processing library. This involves tasks like tokenization, lemmatization, and removing stop words to prepare the text for topic modeling.
2. Topic Modeling and Segmentation:

Subject Extraction: The script identifies the main subjects discussed in the transcript using spaCy's dependency parsing capabilities.
Topic Modeling: It performs Latent Dirichlet Allocation (LDA) using gensim, a topic modeling library. This identifies latent topics within the text based on the co-occurrence of words.
Segment Identification: The script analyzes the dominant topic for each sentence in the transcript and groups consecutive sentences with the same dominant topic into segments.
3. Metadata Generation and Analysis:

Metadata Extraction: For each segment, the script extracts metadata such as start and end times, duration, dominant topic, top keywords related to the topic, and the transcript of the segment.
Video Splitting: The script uses the moviepy library to split the original video into segments based on the identified time boundaries.
AI-Powered Analysis (Optional): If you have access to Google's Gemini API, the script can further analyze each segment by generating a description of the main subject matter, key visual elements, and their relation to the transcript.
4. Output:

The script saves the results, including identified topics, segment metadata, and AI-generated analysis (if enabled), to a JSON file.
It also saves the individual video segments as separate files.
Key Features:

Flexibility: The script can process either a video file or a pre-existing transcript in JSON format.
Customizability: You can customize the number of topics for LDA, choose between different transcription APIs, and provide a prompt for Groq transcription.
AI Integration: The optional integration with Google's Gemini allows for a more comprehensive analysis of the video content, combining visual and textual information.
This script provides a powerful tool for automatically analyzing and segmenting video content based on topics, making it useful for applications like video summarization, content organization, and deeper video understanding.
