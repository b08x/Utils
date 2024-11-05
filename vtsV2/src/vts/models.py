from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Union

@dataclass
class Segment:
    """Represents a video segment with associated metadata"""
    id: int
    start_time: float
    end_time: float
    content: str
    topic_id: Optional[int] = None
    keywords: Optional[List[str]] = None
    analysis: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get segment duration in seconds"""
        return self.end_time - self.start_time

    @classmethod
    def from_dict(cls, data: Dict[str, Union[int, float, str]]) -> 'Segment':
        """Create a Segment instance from a dictionary"""
        return cls(
            id=data.get('id', 0),
            start_time=float(data['start_time']),
            end_time=float(data['end_time']),
            content=data['content'],
            topic_id=data.get('topic_id'),
            keywords=data.get('keywords', []),
            analysis=data.get('analysis')
        )

@dataclass
class TranscriptionResult:
    """Holds transcription data and metadata"""
    segments: List[Dict]
    raw_response: Dict
    api_used: str
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "segments": self.segments,
            "raw_response": self.raw_response,
            "api_used": self.api_used,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class VideoProject:
    """Represents a video processing project"""
    id: str
    input_path: Path
    output_dir: Path
    created_at: datetime
    segments: Optional[List[Segment]] = None
    transcription: Optional[TranscriptionResult] = None
    metadata: Optional[Dict] = None

    @property
    def project_dir(self) -> Path:
        """Get the project's working directory"""
        return self.output_dir / self.id

    @property
    def segments_dir(self) -> Path:
        """Get the directory for video segments"""
        return self.project_dir / "segments"

    @property
    def audio_dir(self) -> Path:
        """Get the directory for audio files"""
        return self.project_dir / "audio"

    def ensure_directories(self) -> None:
        """Create all necessary project directories"""
        for directory in [self.project_dir, self.segments_dir, self.audio_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_segment_path(self, segment_id: int) -> Path:
        """Get the path for a specific segment's video file"""
        return self.segments_dir / f"segment_{segment_id}.mp4"

    def to_dict(self) -> Dict:
        """Convert project to dictionary format"""
        return {
            "id": self.id,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "created_at": self.created_at.isoformat(),
            "segments": [vars(s) for s in (self.segments or [])],
            "transcription": self.transcription.to_dict() if self.transcription else None,
            "metadata": self.metadata
        }