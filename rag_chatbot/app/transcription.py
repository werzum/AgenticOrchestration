"""Transcription workflow orchestration."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_chatbot.agents import (
    SpeechToTextAgent,
    SpeechToTextInputSchema,
    SpeechToTextOutputSchema,
    TaskingInputSchema,
    tasking_agent,
)
from rag_chatbot.app.chat import display_welcome

__all__ = ["transcription_loop"]

console = Console()



def transcription_loop(stt_agent: SpeechToTextAgent) -> None:
    """Main transcription loop for audio-to-text processing."""
    display_welcome()

    while True:
        try:
            audio_path = console.input("The path of the file you want to transcribe: ").strip()
            if not audio_path:
                console.print("[dim]Provide an audio file path or type 'exit'.[/dim]")
                continue

            if audio_path.lower() == "exit":
                console.print("Goodbye! Thanks for using the RAG Chatbot.")
                break

            console.print("\n" + "─" * 80)
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                input_data = SpeechToTextInputSchema(audio_file_path=audio_path)

                task = progress.add_task("[cyan]Transcribing audio...", total=None)
                response: SpeechToTextOutputSchema = stt_agent.execute(input_data)
                progress.remove_task(task)

                console.print(Panel(response.transcription, title="[bold]Transcription[/bold]", border_style="green"))

                task = progress.add_task("[cyan]Structuring tasking command...", total=None)
                structured_command = tasking_agent.run(TaskingInputSchema(input_string=response.transcription))
                progress.remove_task(task)

                console.print(
                    Panel(
                        str(structured_command),
                        title="[bold]Structured Command[/bold]",
                        border_style="blue",
                    )
                )

            console.print("\n" + "─" * 80)

        except Exception as exc:  # pragma: no cover - console app flow
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            console.print("[dim]Please try again or type 'exit' to quit.[/dim]")
