from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

from vts.models import Segment, Topic, VideoProject
from vts.config import Settings

logger = logging.getLogger(__name__)

class MetadataManager:
    """Manages metadata generation and handling for video analysis"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def create_segment_metadata(
        self,
        segment: Segment,
        analysis_results: Optional[Dict] = None
    ) -> Dict:
        """Create metadata for a single segment"""
        metadata = {
            "id": segment.id,
            "timestamp": {
                "start": segment.start_time,
                "end": segment.end_time,
                "duration": str(segment.duration)
            },
            "content": {
                "transcript": segment.content,
                "topic_id": segment.topic_id,
                "keywords": segment.keywords
            },
            "analysis": {
                "confidence": analysis_results.get("confidence", 0.0) if analysis_results else 0.0,
                "visual_description": segment.analysis if segment.analysis else "",
                "entities": analysis_results.get("entities", []) if analysis_results else []
            }
        }
        
        return metadata
    
    def create_topic_metadata(self, topic: Topic) -> Dict:
        """Create metadata for a single topic"""
        return {
            "id": topic.id,
            "keywords": topic.keywords,
            "weight": topic.weight,
            "segment_count": len(topic.segments) if topic.segments else 0,
            "total_duration": str(sum(
                (s.duration for s in topic.segments),
                start=datetime.timedelta(0)
            )) if topic.segments else "0:00:00"
        }
    
    def create_project_metadata(self, project: VideoProject) -> Dict:
        """Create comprehensive project metadata"""
        return {
            "project": {
                "id": project.id,
                "created_at": project.created_at.isoformat(),
                "input_file": str(project.input_path),
                "output_directory": str(project.output_dir)
            },
            "analysis": {
                "total_segments": len(project.segments) if project.segments else 0,
                "total_topics": project.metadata.get("topic_count", 0) if project.metadata else 0,
                "transcription_api": project.metadata.get("transcription_api", "unknown") if project.metadata else "unknown",
                "processing_time": project.metadata.get("processing_time", "unknown") if project.metadata else "unknown"
            },
            "settings": {
                "sample_rate": self.settings.SAMPLE_RATE,
                "num_topics": self.settings.DEFAULT_NUM_TOPICS,
                "min_topic_coherence": self.settings.MIN_TOPIC_COHERENCE
            }
        }
        
    def save_metadata(self, metadata: Dict, output_path: Path) -> None:
        """Save metadata to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise
            
    def load_metadata(self, metadata_path: Path) -> Dict:
        """Load metadata from JSON file"""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise