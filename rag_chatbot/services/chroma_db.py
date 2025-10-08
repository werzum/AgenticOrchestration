"""ChromaDB service abstractions used across the application."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypedDict
from uuid import uuid4

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from rag_chatbot.config import CHUNK_OVERLAP, CHUNK_SIZE


class QueryResult(TypedDict):
    documents: List[str]
    metadatas: List[Dict[str, str]]
    distances: List[float]
    ids: List[str]


class ChromaDBService:
    """Service for interacting with ChromaDB using Ollama embeddings."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path = "./chroma_db",
        recreate_collection: bool = False,
    ) -> None:
        persist_path = Path(persist_directory)
        self.embedding_function = OllamaEmbeddingFunction(timeout=30)

        if recreate_collection and persist_path.exists():
            shutil.rmtree(persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, str]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid4()) for _ in documents]

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, str]] = None,
    ) -> QueryResult:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=max(1, min(n_results, self.get_count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        name_to_delete = collection_name if collection_name is not None else self.collection.name
        self.client.delete_collection(name=name_to_delete)

    def get_count(self) -> int:
        return self.collection.count()

    def delete_by_ids(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def _chunk_text(
        self,
        text: str,
        *,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> Iterable[str]:
        if not text:
            return []

        normalized_chunk_size = max(1, chunk_size)
        normalized_overlap = max(0, min(chunk_overlap, normalized_chunk_size - 1))
        step = normalized_chunk_size - normalized_overlap
        if step <= 0:
            step = normalized_chunk_size

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + normalized_chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        return chunks

    def add_markdown_files_to_collection(self, folder_path: str | Path) -> List[str]:
        folder = Path(folder_path)
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []

        for file_path in folder.rglob("*.md"):
            file_content = file_path.read_text(encoding="utf-8")
            for chunk_idx, chunk in enumerate(self._chunk_text(file_content), start=1):
                documents.append(chunk)
                metadatas.append({
                    "source": str(file_path),
                    "chunk_index": chunk_idx,
                })
                ids.append(str(uuid4()))

        if not documents:
            return []

        return self.add_documents(documents=documents, metadatas=metadatas, ids=ids)
