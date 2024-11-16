from setuptools import setup, find_packages

setup(
    name="video_topic_splitter",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "python-dotenv",
        "deepgram-sdk",
        "groq",
        "pydub",
        "scipy",
        "spacy",
        "gensim",
        "videogrep",
        "moviepy",
        "progressbar2",
        "google-generativeai",
        "Pillow"
    ],
    entry_points={
        'console_scripts': [
            'video-topic-splitter=video_topic_splitter.cli:main',
        ],
    },
    python_requires=">=3.8",
)