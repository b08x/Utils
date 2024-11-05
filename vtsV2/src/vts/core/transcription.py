from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from deepgram import DeepgramClient, PrerecordedOptions
from groq import Groq

from vts.config import Settings
from vts.models import TranscriptionResult

class TranscriptionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.deepgram = DeepgramClient(settings.DEEPGRAM_API_KEY) if settings.DEEPGRAM_API_KEY else None
        self.groq = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    def transcribe_with_deepgram(self, audio_path: Path) -> TranscriptionResult:
        if not self.deepgram:
            raise ValueError("Deepgram API key not configured")

        options = PrerecordedOptions(
            model="nova-2",
            language="en",
            topics=True,
            smart_format=True,
            punctuate=True,
            paragraphs=True,
            utterances=True
        )

        with open(audio_path, "rb") as audio:
            response = self.deepgram.listen.rest.v("1").transcribe_file(
                {"buffer": audio.read(), "mimetype": "audio/wav"},
                options
            )
        
        result = response.to_dict()
        segments = [
            {
                "content": u["transcript"],
                "start": u["start"],
                "end": u["end"]
            }
            for u in result["results"]["utterances"]
        ]

        return TranscriptionResult(
            segments=segments,
            raw_response=result,
            timestamp=datetime.now()
        )

    def transcribe_with_groq(
        self, 
        audio_path: Path, 
        prompt: Optional[str] = None
    ) -> TranscriptionResult:
        if not self.groq:
            raise ValueError("Groq API key not configured")

        with open(audio_path, "rb") as audio:
            response = self.groq.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3",
                prompt=prompt,
                response_format="verbose_json",
                language="en"
            )
        
        result = response.to_dict()
        segments = [
            {
                "content": s["text"],
                "start": s["start"],
                "end": s["end"]
            }
            for s in result["segments"]
        ]

        return TranscriptionResult(
            segments=segments,
            raw_response=result,
            timestamp=datetime.now()
        )