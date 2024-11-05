from vts.core.audio import AudioProcessor
from vts.core.metadata import MetadataManager
from vts.core.segmentation import VideoSegmenter
from vts.core.transcription import TranscriptionManager
from vts.analysis.topics import TopicAnalyzer
from vts.analysis.video import VideoAnalyzer
from vts.models import VideoProject, Segment, TranscriptionResult
from vts.config import Settings

__version__ = "0.2.0"