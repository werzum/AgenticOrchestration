"""Global configuration for the RAG chatbot stack."""

from __future__ import annotations

from pathlib import Path
from typing import Final, FrozenSet

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENT_PATH = Path.home() / "Documents" / "KnowledgeBase"
CHROMA_DIRECTORY = REPO_ROOT / "chroma_db"


class ChatConfig:
    """Convenience namespace for chat-related settings."""

    model: Final[str] = "PetrosStav/gemma3-tools:4b"
    exit_commands: FrozenSet[str] = frozenset({"/exit", "exit", "quit", "/quit"})

    def __new__(cls, *args, **kwargs) -> "ChatConfig":  # pragma: no cover - safeguard only
        raise TypeError("ChatConfig is not meant to be instantiated")


EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
CHUNK_SIZE: Final[int] = 1000
CHUNK_OVERLAP: Final[int] = 200
NUM_CHUNKS_TO_RETRIEVE: Final[int] = 3
SIMILARITY_METRIC: Final[str] = "cosine"
CHROMA_PERSIST_DIR: Final[str] = str(CHROMA_DIRECTORY)
MEMORY_SIZE: Final[int] = 10
MAX_CONTEXT_LENGTH: Final[int] = 4000
