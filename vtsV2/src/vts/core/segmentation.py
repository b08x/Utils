from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import librosa

from vts.models import Segment, VideoProject
from vts.config import Settings

logger = logging.getLogger(__name__)

class VideoSegmenter:
    """Handles video segmentation based on topic changes and audio analysis"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def detect_scene_changes(
        self,
        video_path: Path,
        threshold: float = 0.3,
        min_scene_duration: float = 2.0
    ) -> List[float]:
        """Detect major visual scene changes in the video"""
        logger.info("Detecting scene changes...")
        
        try:
            video = VideoFileClip(str(video_path))
            fps = video.fps
            duration = video.duration
            
            # Sample frames at regular intervals
            frame_times = np.arange(0, duration, 1/fps)
            scenes = []
            last_frame = None
            last_scene_time = 0
            
            for time in frame_times:
                frame = video.get_frame(time)
                
                if last_frame is not None:
                    # Calculate frame difference
                    diff = np.mean(np.abs(frame - last_frame))
                    
                    # If difference exceeds threshold and minimum duration has passed
                    if diff > threshold and (time - last_scene_time) >= min_scene_duration:
                        scenes.append(time)
                        last_scene_time = time
                        
                last_frame = frame
                
            video.close()
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scene changes: {str(e)}")
            raise
            
    def detect_audio_segments(
        self,
        audio_path: Path,
        min_silence_duration: float = 1.0,
        silence_threshold: float = -40
    ) -> List[Tuple[float, float]]:
        """Detect segments based on audio characteristics"""
        logger.info("Detecting audio segments...")
        
        try:
            # Load audio file
            audio = AudioSegment.from_file(str(audio_path))
            
            # Convert to numpy array for processing
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate
            
            # Calculate energy
            frame_length = int(sample_rate * 0.025)  # 25ms frames
            hop_length = int(sample_rate * 0.010)    # 10ms hop
            
            rms = librosa.feature.rms(
                y=samples.astype(float),
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Find silence boundaries
            is_silence = rms < (10 ** (silence_threshold / 20))
            
            # Group continuous silence frames
            silence_regions = []
            start = None
            
            for i, silent in enumerate(is_silence):
                time = i * hop_length / sample_rate
                
                if silent and start is None:
                    start = time
                elif not silent and start is not None:
                    if (time - start) >= min_silence_duration:
                        silence_regions.append((start, time))
                    start = None
                    
            return silence_regions
            
        except Exception as e:
            logger.error(f"Error detecting audio segments: {str(e)}")
            raise
            
    def create_segments(
        self,
        project: VideoProject,
        scene_changes: List[float],
        silence_regions: List[Tuple[float, float]],
        transcription_segments: List[Dict]
    ) -> List[Segment]:
        """Create final segments by combining visual, audio, and transcription data"""
        logger.info("Creating final segments...")
        
        # Combine all potential segment boundaries
        boundaries = set()
        boundaries.add(0)  # Start of video
        
        # Add scene changes
        boundaries.update(scene_changes)
        
        # Add silence boundaries
        for start, end in silence_regions:
            boundaries.add(start)
            boundaries.add(end)
            
        # Add transcription segment boundaries
        for seg in transcription_segments:
            boundaries.add(seg["start"])
            boundaries.add(seg["end"])
            
        # Sort boundaries
        boundaries = sorted(list(boundaries))
        
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            
            # Find transcription content for this segment
            content = []
            for seg in transcription_segments:
                if (seg["start"] >= start_time and seg["start"] < end_time) or \
                   (seg["end"] > start_time and seg["end"] <= end_time):
                    content.append(seg["content"])
                    
            # Create segment only if it has content
            if content:
                segment = Segment(
                    id=len(segments) + 1,
                    start_time=start_time,
                    end_time=end_time,
                    content=" ".join(content)
                )
                segments.append(segment)
                
        return segments
        
    def split_video(
        self,
        project: VideoProject,
        segments: List[Segment],
        output_dir: Path
    ) -> None:
        """Split video into segment files"""
        logger.info("Splitting video into segments...")
        
        try:
            video = VideoFileClip(str(project.input_path))
            
            for segment in segments:
                output_path = output_dir / f"segment_{segment.id}.mp4"
                
                # Extract segment
                clip = video.subclip(segment.start_time, segment.end_time)
                
                # Write segment
                clip.write_videofile(
                    str(output_path),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None  # Suppress moviepy logging
                )
                
            video.close()
            logger.info("Video splitting complete")
            
        except Exception as e:
            logger.error(f"Error splitting video: {str(e)}")
            raise
            
    def process_video(
        self,
        project: VideoProject,
        transcription_segments: List[Dict],
        output_dir: Optional[Path] = None
    ) -> List[Segment]:
        """Main method to process video and create segments"""
        try:
            # Detect scene changes
            scene_changes = self.detect_scene_changes(project.input_path)
            
            # Detect audio segments
            audio_path = project.project_dir / "audio" / "processed_audio.m4a"
            silence_regions = self.detect_audio_segments(audio_path)
            
            # Create segments
            segments = self.create_segments(
                project,
                scene_changes,
                silence_regions,
                transcription_segments
            )
            
            # Split video if output directory is provided
            if output_dir:
                self.split_video(project, segments, output_dir)
                
            return segments
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise