"""Standalone CLI for experimenting with Whisper-based transcription."""

from __future__ import annotations

import instructor
from rich.console import Console
from pydantic import Field

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema

from rag_chatbot.services.transcription import CPPWhisperTranscriptionService


class SpeechToTextInputSchema(BaseIOSchema):
    """Input schema for the Speech-to-Text agent."""

    audio_file_path: str = Field(..., description="Path to the MP3 audio file")


class SpeechToTextOutputSchema(BaseIOSchema):
    """Output schema for the Speech-to-Text agent."""

    transcription: str = Field(..., description="Transcribed text from the audio file")


stt_config = AgentConfig(
    client=instructor.from_provider("ollama/llama3.2", mode=instructor.Mode.JSON),
    model="whisper-base",
    input_schema=SpeechToTextInputSchema,
    output_schema=SpeechToTextOutputSchema,
)


class STTAgent(AtomicAgent):
    """Simple passthrough agent for Whisper CLI experiments."""

    def execute(self, input: SpeechToTextInputSchema, transcription_service) -> SpeechToTextOutputSchema:
        transcription = transcription_service.transcribe(input.audio_file_path)
        return SpeechToTextOutputSchema(transcription=transcription)


if __name__ == "__main__":  # pragma: no cover - ad-hoc CLI
    console = Console()
    stt_agent = STTAgent(config=stt_config)
    transcription_service = CPPWhisperTranscriptionService()

    while True:
        try:
            audio_path = console.input("\nEnter audio file path: ").strip()
            if not audio_path:
                console.print("No audio file path provided.")
                continue

            input_data = SpeechToTextInputSchema(audio_file_path=audio_path)
            result = stt_agent.execute(input_data, transcription_service)
            result.transcription.replace(",", "")

            console.print("\nTranscription Result:")
            console.print(result)

        except KeyboardInterrupt:
            console.print("\n[bold]Goodbye! Thanks for using the RAG Chatbot.[/bold]")
            break
        except Exception as exc:  # pragma: no cover - console app
            console.print(f"\n[bold red]Fatal error:[/bold red] {exc}")
