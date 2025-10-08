"""Transcription services backed by Whisper models."""

from __future__ import annotations
import whisper

class WhisperTranscriptionService:
    """Python Whisper transcription helper."""

    def __init__(self, model_name: str = "base", device: str = "cpu") -> None:
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_file_path: str, language: str) -> str:
        result = self.model.transcribe(audio_file_path, language=language)
        return result["text"]
