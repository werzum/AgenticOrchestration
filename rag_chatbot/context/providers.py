"""Context provider definitions used by the system prompt generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


@dataclass
class ChunkItem:
    content: str
    metadata: Dict[str, object] = field(default_factory=dict)


class RAGContextProvider(SystemPromptContextProviderBase):
    """Simple in-memory context provider storing retrieved chunks."""

    def __init__(self, title: str) -> None:
        super().__init__(title=title)
        self.chunks: List[ChunkItem] = []

    def get_info(self) -> str:
        return "\n\n".join(
            f"Chunk {idx}:\nMetadata: {item.metadata}\nContent:\n{item.content}\n{'-' * 80}"
            for idx, item in enumerate(self.chunks, 1)
        )
