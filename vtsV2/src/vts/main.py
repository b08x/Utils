import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from vts.config import Settings
from vts.core.audio import AudioProcessor
from vts.core.transcription import TranscriptionManager
from vts.core.reporting import ReportGenerator
from vts.analysis.topics import TopicAnalyzer
from vts.analysis.video import VideoAnalyzer
from vts.models import VideoProject, AnalysisReport
from vts.utils import setup_logging

logger = logging.getLogger(__name__)
console = Console()

class VideoTopicSegmentation:
    """Main application class for Video Topic Segmentation"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.audio_processor = AudioProcessor(self.settings)
        self.transcription_manager = TranscriptionManager(self.settings)
        self.topic_analyzer = TopicAnalyzer(self.settings)
        self.video_analyzer = VideoAnalyzer(self.settings)
        self.report_generator = ReportGenerator()
        
    def create_project(self, input_path: Path) -> VideoProject:
        """Create a new project instance"""
        project_id = f"{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_dir = self.settings.BASE_OUTPUT_DIR / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        return VideoProject(
            id=project_id,
            input_path=input_path,
            output_dir=self.settings.BASE_OUTPUT_DIR,
            created_at=datetime.now()
        )

    def process_audio(self, project: VideoProject) -> Path:
        """Process audio from video file"""
        logger.info("Processing audio...")
        audio_dir = project.project_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Extract and normalize audio
        raw_audio = audio_dir / "raw_audio.wav"
        self.audio_processor.extract_audio(project.input_path, raw_audio)
        
        normalized_audio = audio_dir / "normalized_audio.wav"
        result = self.audio_processor.normalize_audio(raw_audio, normalized_audio)
        if result["status"] == "error":
            raise RuntimeError(f"Audio normalization failed: {result['message']}")
            
        # Convert to mono and resample
        processed_audio = audio_dir / "processed_audio.m4a"
        result = self.audio_processor.convert_to_mono(normalized_audio, processed_audio)
        if result["status"] == "error":
            raise RuntimeError(f"Audio conversion failed: {result['message']}")
            
        return processed_audio

    def transcribe(self, audio_path: Path, api: str = "deepgram", prompt: Optional[str] = None):
        """Transcribe audio using specified API"""
        logger.info(f"Transcribing audio using {api}...")
        
        if api == "deepgram":
            return self.transcription_manager.transcribe_with_deepgram(audio_path)
        elif api == "groq":
            return self.transcription_manager.transcribe_with_groq(audio_path, prompt)
        else:
            raise ValueError(f"Unsupported transcription API: {api}")

    def analyze_topics(self, project: VideoProject):
        """Perform topic analysis on transcribed content"""
        logger.info("Analyzing topics...")
        
        # Combine all transcript segments
        full_text = " ".join(seg["content"] for seg in project.transcription.segments)
        
        # Preprocess and build topic model
        subjects = self.topic_analyzer.preprocess_text(full_text)
        model, dictionary = self.topic_analyzer.build_topic_model(subjects)
        
        # Analyze segments
        segments = []
        for i, seg in enumerate(project.transcription.segments):
            topic_id, confidence = self.topic_analyzer.get_dominant_topic(
                seg["content"], model, dictionary
            )
            # Create Segment object instead of dictionary
            segment = Segment(
                id=i + 1,
                start_time=float(seg["start"]),  # Convert to float
                end_time=float(seg["end"]),      # Convert to float
                content=seg["content"],
                topic_id=topic_id,
                keywords=None  # Will be populated later
            )
            segments.append(segment)
            
        return model, segments

    def analyze_video(self, project: VideoProject, segments):
        """Perform video analysis on segments"""
        logger.info("Analyzing video segments...")
        segments_dir = project.project_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        # Ensure segments are Segment objects
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
        
        return self.video_analyzer.split_video(project, segments)

    def generate_report(self, project: VideoProject, analysis_results: Dict[str, Any]) -> Path:
        """Generate analysis report"""
        logger.info("Generating analysis report...")
        report = create_analysis_report(analysis_results, project.project_dir)
        return self.report_generator.save_report(report, project.project_dir)

    def process_video(
        self,
        input_path: Path,
        api: str = "deepgram",
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Create project
                task = progress.add_task("Creating project...", total=None)
                project = self.create_project(input_path)
                progress.update(task, completed=True)

                # Process audio
                task = progress.add_task("Processing audio...", total=None)
                processed_audio = self.process_audio(project)
                progress.update(task, completed=True)

                # Transcribe
                task = progress.add_task(f"Transcribing with {api}...", total=None)
                project.transcription = self.transcribe(processed_audio, api, prompt)
                progress.update(task, completed=True)

                # Analyze topics
                task = progress.add_task("Analyzing topics...", total=None)
                topic_model, segments = self.analyze_topics(project)
                progress.update(task, completed=True)

                # Analyze video
                task = progress.add_task("Analyzing video segments...", total=None)
                analyzed_segments = self.analyze_video(project, segments)
                progress.update(task, completed=True)

                # Generate report
                task = progress.add_task("Generating report...", total=None)
                analysis_results = {
                    "project_id": project.id,
                    "topics": topic_model.show_topics(),
                    "segments": analyzed_segments,
                    "metadata": {
                        "input_file": str(input_path),
                        "created_at": datetime.now().isoformat(),
                        "transcription_api": api
                    }
                }
                report_path = self.generate_report(project, analysis_results)
                progress.update(task, completed=True)

                return {
                    "project_id": project.id,
                    "project_dir": str(project.project_dir),
                    "report_path": str(report_path),
                    "results": analysis_results
                }

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise

@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input video file path"
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Base output directory"
)
@click.option(
    "--api",
    type=click.Choice(["deepgram", "groq"]),
    default="deepgram",
    help="Transcription API to use"
)
@click.option(
    "--prompt",
    help="Optional prompt for Groq transcription"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging"
)
def main(
    input: Path,
    output: Optional[Path],
    api: str,
    prompt: Optional[str],
    debug: bool
):
    """Video Topic Segmentation - Analyze and segment videos based on content topics"""
    try:
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        setup_logging(log_level)
        
        # Initialize settings
        settings = Settings()
        if output:
            settings.BASE_OUTPUT_DIR = output
            
        # Process video
        vts = VideoTopicSegmentation(settings)
        results = vts.process_video(input, api, prompt)
        
        # Print results
        console.print("\n✨ Processing complete!")
        console.print(f"Project ID: {results['project_id']}")
        console.print(f"Project directory: {results['project_dir']}")
        console.print(f"Analysis report: {results['report_path']}")
        
    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}", style="bold red")
        if debug:
            console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    main()