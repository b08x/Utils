# Video Topic Segmentation (VTS)

VTS is a powerful Python application that automatically segments and analyzes video content based on topics. It combines audio processing, speech recognition, natural language processing, and computer vision to provide detailed insights into video content.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Automated Video Segmentation**: Intelligently splits videos into coherent segments based on topic changes
- **Multi-Modal Analysis**: Combines audio, text, and visual analysis for comprehensive understanding
- **Topic Modeling**: Identifies and clusters main topics discussed in the video
- **Advanced Audio Processing**:
  - Noise reduction and normalization
  - Silence removal
  - High-pass filtering
  - Automated gain control
- **Multiple Transcription Options**:
  - Deepgram API integration
  - Groq API support
- **Visual Analysis**: Uses Google's Gemini to analyze visual content
- **Detailed Reports**: Generates comprehensive markdown reports including:
  - Topic summaries
  - Segment analysis
  - Timestamps
  - Content transcriptions
  - Visual scene descriptions

## 📋 Prerequisites

- Python 3.9 or higher
- FFmpeg
- Required API keys:
  - Deepgram and/or Groq for transcription
  - Google Gemini for visual analysis

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vts.git
cd vts
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

## 🎯 Quick Start

1. Basic usage:
```bash
python -m vts.main -i path/to/video.mp4
```

2. With custom options:
```bash
python -m vts.main -i video.mp4 -o output/dir --api groq --debug
```

3. Using as a Python package:
```python
from vts import VideoTopicSegmentation
from vts.config import Settings

# Initialize
settings = Settings()
vts = VideoTopicSegmentation(settings)

# Process video
results = vts.process_video("video.mp4", api="deepgram")
print(f"Analysis report generated at: {results['report_path']}")
```

## ⚙️ Configuration

Create a `.env` file with your settings:

```env
# API Keys
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key

# Audio Processing
SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_BITRATE=128k
HIGHPASS_FREQ=200
LOWPASS_FREQ=8000

# Topic Modeling
DEFAULT_NUM_TOPICS=5
MIN_TOPIC_COHERENCE=0.3

# Output
BASE_OUTPUT_DIR=./output
```

## 📝 Example Output

The application generates a detailed markdown report for each processed video:

```markdown
# Video Analysis Report: Example Video

## Overview
- Analysis Date: 2024-10-22 14:30:00
- Total Duration: 00:15:30
- Number of Segments: 8
- Number of Topics: 3

## Topics Identified
### Topic 1
- Keywords: technology, AI, future, innovation
- Segments: 3

[... detailed segment analysis ...]
```

## 🛠️ Development

1. Set up development environment:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest tests/
```

3. Format code:
```bash
black src/
isort src/
```

## 📊 Project Structure

```
vts/
├── pyproject.toml
├── README.md
├── src/
│   └── vts/
│       ├── core/
│       │   ├── audio.py
│       │   ├── metadata.py
│       │   ├── reporting.py
│       │   └── transcription.py
│       ├── analysis/
│       │   ├── topics.py
│       │   └── video.py
│       ├── config.py
│       ├── models.py
│       └── utils.py
└── tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- [Deepgram](https://deepgram.com/) for speech recognition
- [Groq](https://groq.com/) for alternative transcription
- [Google Gemini](https://ai.google/discover/gemini/) for visual analysis
- [spaCy](https://spacy.io/) for NLP capabilities
- [Gensim](https://radimrehurek.com/gensim/) for topic modeling

## 📮 Contact

Your Name - [@yourusername](https://twitter.com/yourusername)

Project Link: [https://github.com/yourusername/vts](https://github.com/yourusername/vts)