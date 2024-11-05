from pathlib import Path
import logging
from typing import List, Dict, Union, Optional
import google.generativeai as genai
from PIL import Image
from moviepy.editor import VideoFileClip

from vts.config import Settings
from vts.models import Segment, VideoProject

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            logger.warning("No GEMINI_API_KEY provided. Visual analysis will be disabled.")
            self.model = None

    def split_video(
        self, 
        project: VideoProject,
        segments: Union[List[Segment], List[Dict]]
    ) -> List[Dict]:
        """Split video into segments and analyze each one"""
        logger.info(f"Splitting video into {len(segments)} segments...")
        
        # Convert dictionaries to Segment objects if necessary
        if segments and isinstance(segments[0], dict):
            segments = [
                Segment(
                    id=s.get("id", i+1),
                    start_time=float(s["start_time"]),
                    end_time=float(s["end_time"]),
                    content=s["content"],
                    topic_id=s.get("topic_id"),
                    keywords=s.get("keywords", [])
                )
                for i, s in enumerate(segments)
            ]
        
        try:
            video = VideoFileClip(str(project.input_path))
            analyzed_segments = []

            for segment in segments:
                try:
                    # Create the segment video
                    logger.debug(f"Processing segment {segment.id} ({segment.start_time} - {segment.end_time})")
                    clip = video.subclip(segment.start_time, segment.end_time)
                    output_path = project.project_dir / "segments" / f"segment_{segment.id}.mp4"
                    
                    # Ensure parent directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    clip.write_videofile(
                        str(output_path), 
                        codec="libx264", 
                        audio_codec="aac",
                        logger=None  # Suppress moviepy output
                    )
                    
                    # Analyze the segment if Gemini is available
                    analysis = None
                    if self.model:
                        analysis = self.analyze_segment(segment, output_path)
                    
                    # Add to results
                    analyzed_segments.append({
                        "segment_id": segment.id,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "transcript": segment.content,
                        "topic": segment.topic_id,
                        "keywords": segment.keywords,
                        "gemini_analysis": analysis
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing segment {segment.id}: {str(e)}")
                    # Continue with next segment instead of failing completely
                    continue

            video.close()
            return analyzed_segments
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def analyze_segment(self, segment: Segment, video_path: Path) -> Optional[str]:
        """Analyze a single video segment using Gemini"""
        if not self.model:
            return None
            
        try:
            video = VideoFileClip(str(video_path))
            frame = video.get_frame(0)  # Get first frame
            image = Image.fromarray(frame)
            video.close()

            prompt = (
                f"Analyze this video segment. The transcript is: '{segment.content}'. "
                f"Describe the main subject matter, key visual elements, and how they "
                f"relate to the transcript."
            )

            response = self.model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing segment {segment.id}: {str(e)}")
            return None

    def get_segment_thumbnail(self, video_path: Path, time: float) -> Optional[Image.Image]:
        """Extract thumbnail from video at specified time"""
        try:
            video = VideoFileClip(str(video_path))
            frame = video.get_frame(time)
            video.close()
            return Image.fromarray(frame)
        except Exception as e:
            logger.error(f"Error extracting thumbnail: {str(e)}")
            return None

    def process_segment(
        self,
        segment: Segment,
        input_video: Path,
        output_path: Path,
        analyze: bool = True
    ) -> Dict:
        """Process a single segment with all steps"""
        try:
            # Extract segment video
            video = VideoFileClip(str(input_video))
            clip = video.subclip(segment.start_time, segment.end_time)
            clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            video.close()

            # Get thumbnail
            thumbnail = self.get_segment_thumbnail(output_path, 0)
            
            # Analyze if requested
            analysis = None
            if analyze and self.model and thumbnail:
                analysis = self.analyze_segment(segment, output_path)

            return {
                "segment_id": segment.id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.start_time,
                "transcript": segment.content,
                "topic": segment.topic_id,
                "keywords": segment.keywords,
                "analysis": analysis,
                "output_path": str(output_path)
            }

        except Exception as e:
            logger.error(f"Error processing segment {segment.id}: {str(e)}")
            raise