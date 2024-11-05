from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    DEEPGRAM_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Audio Processing
    SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_BITRATE: str = "128k"
    HIGHPASS_FREQ: int = 200
    LOWPASS_FREQ: int = 8000
    SILENCE_THRESHOLD: float = -40.0
    MIN_SILENCE_DURATION: float = 1.0
    NORMALIZE_TARGET_LEVEL: float = -23.0

    # Topic Modeling
    DEFAULT_NUM_TOPICS: int = 5
    MIN_TOPIC_COHERENCE: float = 0.3
    MAX_TOPIC_ITERATIONS: int = 400
    NUM_TOPIC_WORDS: int = 10

    # Video Processing
    MIN_SEGMENT_DURATION: float = 1.0
    MAX_SEGMENT_DURATION: float = 300.0  # 5 minutes
    VIDEO_CODEC: str = "libx264"
    AUDIO_CODEC: str = "aac"
    VIDEO_BITRATE: str = "2M"
    
    # File Paths
    BASE_OUTPUT_DIR: Path = Path.cwd() / "output"
    CACHE_DIR: Path = Path.cwd() / "cache"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("BASE_OUTPUT_DIR", "CACHE_DIR")
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def get_api_key(self, api: str) -> Optional[str]:
        """Get API key by service name"""
        api_keys = {
            "deepgram": self.DEEPGRAM_API_KEY,
            "groq": self.GROQ_API_KEY,
            "gemini": self.GEMINI_API_KEY
        }
        return api_keys.get(api.lower())

    def validate_api_keys(self, required_apis: list[str]) -> None:
        """Validate that required API keys are present"""
        missing_keys = [
            api for api in required_apis 
            if not self.get_api_key(api)
        ]
        if missing_keys:
            raise ValueError(
                f"Missing required API keys for: {', '.join(missing_keys)}"
            )