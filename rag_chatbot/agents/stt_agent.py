"""Agent responsible for Whisper-based speech transcription."""

from __future__ import annotations

import instructor
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from pydantic import Field

from rag_chatbot.services import WhisperTranscriptionService


class SpeechToTextInputSchema(BaseIOSchema):
    """Input schema for the Speech-to-Text agent."""

    audio_file_path: str = Field(..., description="Path to the audio file to transcribe")


class SpeechToTextOutputSchema(BaseIOSchema):
    """Output schema for the Speech-to-Text agent."""

    transcription: str = Field(..., description="Transcribed text from the audio file")


stt_config = BaseAgentConfig(
    client=instructor.from_provider("ollama/llama3.2", mode=instructor.Mode.JSON),
    model="whisper-base",
    input_schema=SpeechToTextInputSchema,
    output_schema=SpeechToTextOutputSchema,
)



class SpeechToTextAgent(BaseAgent):
    """Agent wrapper that delegates to a Whisper transcription service."""

    def execute(self, input: SpeechToTextInputSchema) -> SpeechToTextOutputSchema:
        transcription_service = WhisperTranscriptionService(model_name="base", device="cuda")
        transcription = transcription_service.transcribe(
            input.audio_file_path,
            language="en",
        )
        return SpeechToTextOutputSchema(transcription=transcription)
