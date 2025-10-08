"""Command line entry point for the RAG chatbot application."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_chatbot.agents import SpeechToTextAgent, stt_config
from rag_chatbot.app import OrchestrationEnvironment, chat_loop, initialize_system, transcription_loop
from rag_chatbot.config import DEFAULT_DOCUMENT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG chatbot utilities")
    parser.add_argument(
        "--mode",
        choices={"chat", "transcribe"},
        default="chat",
        help="Select chat interface or audio transcription pipeline",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENT_PATH,
        help="Path containing markdown files to ingest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "chat":
        environment: OrchestrationEnvironment = initialize_system(str(args.documents))
        chat_loop(environment)
    else:
        stt_agent = SpeechToTextAgent(config=stt_config)
        transcription_loop(stt_agent)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    try:
        main()
    except KeyboardInterrupt:
        pass
