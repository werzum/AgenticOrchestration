"""Chat interface orchestration for the RAG chatbot."""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_chatbot.agents import (
    RAGQuestionAnsweringAgentInputSchema,
    RAGQuestionAnsweringAgentOutputSchema,
    RAGQueryAgentInputSchema,
    RAGQueryAgentOutputSchema,
    qa_agent,
    query_agent,
)
from rag_chatbot.config import CHROMA_PERSIST_DIR, NUM_CHUNKS_TO_RETRIEVE
from rag_chatbot.context import ChunkItem, RAGContextProvider
from rag_chatbot.services import ChromaDBService

__all__ = [
    "initialize_system",
    "chat_loop",
    "display_answer",
    "display_chunks",
    "display_query_info",
    "display_welcome",
]

console = Console()



def initialize_system(document_path: str) -> tuple[ChromaDBService, RAGContextProvider]:
    """Initialize vector store, context provider, and agent wiring."""
    console.print("\n[bold magenta]🚀 Initializing RAG Chatbot System...[/bold magenta]")

    console.print("[dim]• Initializing vector database...[/dim]")
    chroma_db = ChromaDBService(
        collection_name="knowledge_base",
        persist_directory=CHROMA_PERSIST_DIR,
        recreate_collection=False,
    )

    chroma_db.add_markdown_files_to_collection(document_path)

    console.print("[dim]• Creating context provider...[/dim]")
    rag_context = RAGContextProvider("RAG Context")

    console.print("[dim]• Registering context provider with agents...[/dim]")
    query_agent.register_context_provider("rag_context", rag_context)
    qa_agent.register_context_provider("rag_context", rag_context)

    console.print("[bold green]✨ System initialized successfully![/bold green]\n")
    return chroma_db, rag_context



def display_welcome() -> None:
    """Display welcome message and starter questions."""
    welcome_panel = Panel("Chatbot", title="[bold blue]RAG Chatbot[/bold blue]", border_style="blue", padding=(1, 2))
    console.print("\n")
    console.print(welcome_panel)
    console.print("\n" + "─" * 80 + "\n")



def display_chunks(chunks: Iterable[ChunkItem]) -> None:
    """Display the retrieved chunks in a formatted way."""
    console.print("\n[bold cyan]Retrieved Text Chunks:[/bold cyan]")

    for idx, chunk in enumerate(chunks, 1):
        distance = chunk.metadata.get("distance")
        suffix = f" (Distance: {float(distance):.4f})" if isinstance(distance, (float, int)) else ""
        chunk_panel = Panel(
            Markdown(chunk.content),
            title=f"[bold]Chunk {idx}{suffix}[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(chunk_panel)
        console.print()



def display_query_info(query_output: RAGQueryAgentOutputSchema) -> None:
    """Display information about the generated query."""
    query_panel = Panel(
        "[yellow]Generated Query:[/yellow] {query}\n\n[yellow]Reasoning:[/yellow] {reasoning}".format(
            query=query_output.query,
            reasoning=query_output.reasoning,
        ),
        title="[bold]🔍 Semantic Search Strategy[/bold]",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(query_panel)



def display_answer(qa_output: RAGQuestionAnsweringAgentOutputSchema) -> None:
    """Display the reasoning and answer from the QA agent."""
    reasoning_panel = Panel(
        Markdown(qa_output.reasoning),
        title="[bold]Analysis & Reasoning[/bold]",
        border_style="green",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(reasoning_panel)

    answer_panel = Panel(
        Markdown(qa_output.answer),
        title="[bold]Answer[/bold]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(answer_panel)



def chat_loop(chroma_db: ChromaDBService, rag_context: RAGContextProvider) -> None:
    """Main chat loop for the RAG chatbot."""
    display_welcome()

    while True:
        try:
            user_message = console.input("\n[bold blue]Your question:[/bold blue] ").strip()

            if not user_message:
                console.print("[dim]Please enter a question or type 'exit'.[/dim]")
                continue

            if user_message.lower() == "exit":
                console.print("Goodbye! Thanks for using the RAG Chatbot.")
                break

            console.print("\n" + "─" * 80)
            console.print("\n[bold magenta]🔄 Processing your question...[/bold magenta]")

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("[cyan]Generating semantic search query...", total=None)
                query_output = query_agent.run(RAGQueryAgentInputSchema(user_message=user_message))
                progress.remove_task(task)

                display_query_info(query_output)

                task = progress.add_task("[cyan]Searching knowledge base...", total=None)
                search_results = chroma_db.query(query_text=query_output.query, n_results=NUM_CHUNKS_TO_RETRIEVE)
                progress.remove_task(task)

                rag_context.chunks = [
                    ChunkItem(content=doc, metadata={"chunk_id": doc_id, "distance": distance})
                    for doc, doc_id, distance in zip(
                        search_results["documents"],
                        search_results["ids"],
                        search_results["distances"],
                    )
                ]

                display_chunks(rag_context.chunks)

                task = progress.add_task("[cyan]Analyzing chunks and generating answer...", total=None)
                qa_output = qa_agent.run(RAGQuestionAnsweringAgentInputSchema(question=user_message))
                progress.remove_task(task)

                display_answer(qa_output)

            console.print("\n" + "─" * 80)

        except Exception as exc:  # pragma: no cover - console app flow
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            console.print("[dim]Please try again or type 'exit' to quit.[/dim]")
