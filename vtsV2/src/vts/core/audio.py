from pathlib import Path
import subprocess
from typing import Dict, Union
from pydub import AudioSegment

from vts.config import Settings
from vts.models import VideoProject

class AudioProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def convert_to_mono(self, input_path: Path, output_path: Path) -> Dict[str, str]:
        try:
            command = [
                "ffmpeg",
                "-i", str(input_path),
                "-af", f"highpass=f={self.settings.HIGHPASS_FREQ}, acompressor=threshold=-12dB:ratio=4:attack=5:release=50, loudnorm",
                "-ar", str(self.settings.SAMPLE_RATE),
                "-ac", str(self.settings.AUDIO_CHANNELS),
                "-c:a", "aac",
                "-b:a", self.settings.AUDIO_BITRATE,
                str(output_path)
            ]
            subprocess.run(command, check=True, capture_output=True)
            return {"status": "success", "message": f"Audio converted successfully"}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": str(e)}

    def normalize_audio(self, input_path: Path, output_path: Path) -> Dict[str, str]:
        try:
            command = [
                "ffmpeg-normalize",
                "-pr",
                "-tp", "-3.0",
                "-nt", "rms",
                str(input_path),
                "-prf", f"highpass=f={self.settings.HIGHPASS_FREQ}, loudnorm",
                "-prf", "dynaudnorm=p=0.4:s=15",
                "-pof", f"lowpass=f={self.settings.LOWPASS_FREQ}",
                "-ar", str(self.settings.SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                "--keep-loudness-range-target",
                "-o", str(output_path)
            ]
            subprocess.run(command, check=True, capture_output=True)
            return {"status": "success", "message": "Audio normalized successfully"}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": str(e)}

    def extract_audio(self, video_path: Path, output_path: Path) -> None:
        audio = AudioSegment.from_file(str(video_path))
        audio.export(str(output_path), format="wav")