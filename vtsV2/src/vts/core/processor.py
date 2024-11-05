from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import videogrep
from deepgram import DeepgramClient, PrerecordedOptions
from groq import Groq
import google.generativeai as genai
from moviepy.editor import VideoFileClip

from vts.core.audio import AudioProcessor
from vts.analysis.topics import TopicAnalyzer
from vts.models import VideoProject, Segment
from vts.config import Settings

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Main processor class that orchestrates the video analysis pipeline"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.audio_processor = AudioProcessor(settings)
        self.topic_analyzer = TopicAnalyzer(settings)
        
        # Initialize API clients
        self.deepgram = (DeepgramClient(settings.DEEPGRAM_API_KEY) 
                        if settings.DEEPGRAM_API_KEY else None)
        self.groq = (Groq(api_key=settings.GROQ_API_KEY)
                    if settings.GROQ_API_KEY else None)
        
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.gemini = None

    def process_video(
        self, 
        input_path: Path,
        output_dir: Path,
        api: str = "deepgram",
        num_topics: int = 5,
        groq_prompt: Optional[str] = None
    ) -> Dict:
        """Main processing pipeline"""
        logger.info(f"Processing video: {input_path}")
        
        # Create project
        project = self._create_project(input_path, output_dir)
        
        try:
            # Process audio
            audio_path = self._process_audio(project)
            
            # Get transcript
            transcript = self._get_transcript(project, audio_path, api, groq_prompt)
            
            # Perform topic modeling and segmentation
            results = self._analyze_content(project, transcript, num_topics)
            
            # Save results
            self._save_results(project, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def _create_project(self, input_path: Path, output_dir: Path) -> VideoProject:
        """Create and setup project directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{input_path.stem}_{timestamp}"
        
        project = VideoProject(
            id=project_id,
            input_path=input_path,
            output_dir=output_dir,
            created_at=datetime.now()
        )
        
        # Create project directories
        project.project_dir.mkdir(parents=True, exist_ok=True)
        (project.project_dir / "audio").mkdir(exist_ok=True)
        (project.project_dir / "segments").mkdir(exist_ok=True)
        
        return project

    def _process_audio(self, project: VideoProject) -> Path:
        """Process audio from video file"""
        logger.info("Processing audio...")
        
        audio_dir = project.project_dir / "audio"
        
        # Normalize audio
        normalized_path = audio_dir / "normalized_audio.wav"
        self.audio_processor.normalize_audio(project.input_path, normalized_path)
        
        # Remove silence
        unsilenced_path = audio_dir / "unsilenced_audio.wav"
        self.audio_processor.remove_silence(normalized_path, unsilenced_path)
        
        # Convert to mono and resample
        processed_path = audio_dir / "processed_audio.m4a"
        self.audio_processor.convert_to_mono(unsilenced_path, processed_path)
        
        return processed_path

    def _get_transcript(
        self,
        project: VideoProject,
        audio_path: Path,
        api: str,
        groq_prompt: Optional[str]
    ) -> List[Dict]:
        """Get transcript either from video or through transcription"""
        logger.info("Getting transcript...")
        
        # Try to parse embedded transcript
        transcript = videogrep.parse_transcript(str(project.input_path))
        
        if not transcript:
            logger.info("No embedded transcript found. Transcribing audio...")
            
            if api == "deepgram" and self.deepgram:
                transcription = self._transcribe_with_deepgram(audio_path)
                transcript = [
                    {
                        "content": u["transcript"],
                        "start": u["start"],
                        "end": u["end"]
                    }
                    for u in transcription["results"]["utterances"]
                ]
                
            elif api == "groq" and self.groq:
                transcription = self._transcribe_with_groq(audio_path, groq_prompt)
                transcript = [
                    {
                        "content": s["text"],
                        "start": s["start"],
                        "end": s["end"]
                    }
                    for s in transcription["segments"]
                ]
                
            else:
                raise ValueError(f"Invalid or unconfigured API: {api}")
                
            # Save raw transcription
            with open(project.project_dir / "transcription.json", 'w') as f:
                json.dump(transcription, f, indent=2)
        
        # Save formatted transcript
        with open(project.project_dir / "transcript.json", 'w') as f:
            json.dump(transcript, f, indent=2)
            
        return transcript

    def _transcribe_with_deepgram(self, audio_path: Path) -> Dict:
        """Transcribe using Deepgram"""
        options = PrerecordedOptions(
            model="nova-2",
            language="en",
            topics=True,
            smart_format=True,
            punctuate=True,
            paragraphs=True,
            utterances=True
        )
        
        with open(audio_path, 'rb') as audio:
            response = self.deepgram.listen.rest.v1.transcribe_file(
                {"buffer": audio.read(), "mimetype": "audio/m4a"},
                options
            )
        return response.to_dict()

    def _transcribe_with_groq(self, audio_path: Path, prompt: Optional[str]) -> Dict:
        """Transcribe using Groq"""
        with open(audio_path, 'rb') as audio:
            response = self.groq.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3",
                prompt=prompt,
                response_format="verbose_json",
                language="en"
            )
        return json.loads(response.text)

    def _analyze_content(
        self,
        project: VideoProject,
        transcript: List[Dict],
        num_topics: int
    ) -> Dict:
        """Analyze content using topic modeling and generate segments"""
        logger.info("Analyzing content...")
        
        # Combine transcript text
        full_text = " ".join(seg["content"] for seg in transcript)
        
        # Build topic model
        subjects = self.topic_analyzer.preprocess_text(full_text)
        model, dictionary = self.topic_analyzer.build_topic_model(subjects)
        
        # Create segments based on topics
        segments = []
        for i, seg in enumerate(transcript):
            topic_id, confidence = self.topic_analyzer.get_dominant_topic(
                seg["content"], model, dictionary
            )
            segments.append(Segment(
                id=i + 1,
                start_time=float(seg["start"]),
                end_time=float(seg["end"]),
                content=seg["content"],
                topic_id=topic_id
            ))
        
        # Split video and analyze segments
        analyzed_segments = self._analyze_segments(project, segments)
        
        # Generate results
        return {
            "project_id": project.id,
            "topics": [
                {
                    "topic_id": tid,
                    "words": [word for word, _ in model.show_topic(tid, topn=10)]
                }
                for tid in range(num_topics)
            ],
            "segments": analyzed_segments
        }

    def _analyze_segments(
        self,
        project: VideoProject,
        segments: List[Segment]
    ) -> List[Dict]:
        """Split video into segments and analyze each one"""
        logger.info("Analyzing segments...")
        
        video = VideoFileClip(str(project.input_path))
        analyzed_segments = []
        
        for segment in segments:
            try:
                # Extract segment video
                output_path = project.project_dir / "segments" / f"segment_{segment.id}.mp4"
                clip = video.subclip(segment.start_time, segment.end_time)
                clip.write_videofile(
                    str(output_path),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None
                )
                
                # Analyze with Gemini if available
                analysis = None
                if self.gemini:
                    frame = clip.get_frame(0)
                    response = self.gemini.generate_content([
                        f"Analyze this video segment. The transcript is: '{segment.content}'.",
                        Image.fromarray(frame)
                    ])
                    analysis = response.text
                
                analyzed_segments.append({
                    "segment_id": segment.id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "transcript": segment.content,
                    "topic": segment.topic_id,
                    "gemini_analysis": analysis
                })
                
            except Exception as e:
                logger.error(f"Error processing segment {segment.id}: {str(e)}")
                continue
        
        video.close()
        return analyzed_segments

    def _save_results(self, project: VideoProject, results: Dict) -> None:
        """Save analysis results"""
        results_path = project.project_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")