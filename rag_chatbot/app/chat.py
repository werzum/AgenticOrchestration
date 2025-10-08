"""Chat interface orchestration for the RAG chatbot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rich.console import Console

from rag_chatbot.agents import (
    OrchestratorInputSchema,
    OrchestratorOutputSchema,
    RAGQuestionAnsweringAgentInputSchema,
    RAGQuestionAnsweringAgentOutputSchema,
    RAGQueryAgentInputSchema,
    RAGQueryAgentOutputSchema,
    SpeechToTextAgent,
    SpeechToTextInputSchema,
    SpeechToTextOutputSchema,
    TaskingInputSchema,
    TaskingOutputSchema,
    orchestrator_agent,
    qa_agent,
    query_agent,
    stt_config,
    tasking_agent,
)
from rag_chatbot.config import CHROMA_PERSIST_DIR, ChatConfig, NUM_CHUNKS_TO_RETRIEVE
from rag_chatbot.context import ChunkItem, RAGContextProvider
from rag_chatbot.services import ChromaDBService

__all__ = [
    "OrchestrationEnvironment",
    "initialize_system",
    "chat_loop",
    "display_answer",
    "display_chunks",
    "display_query_info",
    "display_transcription",
    "display_tasking",
    "display_welcome",
]


@dataclass
class OrchestrationEnvironment:
    """Runtime wiring for orchestrated chat interactions."""

    chroma_db: ChromaDBService
    rag_context: RAGContextProvider
    stt_agent: SpeechToTextAgent

console = Console()



def initialize_system(document_path: str) -> OrchestrationEnvironment:
    """Initialize vector store, context provider, orchestration agents, and tool wiring."""
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

    console.print("[dim]• Loading speech-to-text agent...[/dim]")
    stt_agent = SpeechToTextAgent(config=stt_config)

    console.print("[bold green]✨ System initialized successfully![/bold green]\n")
    return OrchestrationEnvironment(
        chroma_db=chroma_db,
        rag_context=rag_context,
        stt_agent=stt_agent,
    )



def display_welcome() -> None:
    """Display welcome message and starter questions."""
    console.print("\n[bold blue]RAG Chatbot[/bold blue]")
    console.print("Type 'exit' any time to leave.\n" + "─" * 40)



def display_chunks(chunks: Iterable[ChunkItem]) -> None:
    """Display the retrieved chunks in a formatted way."""
    console.print("\n[bold cyan]Retrieved Text Chunks[/bold cyan]")

    for idx, chunk in enumerate(chunks, 1):
        distance = chunk.metadata.get("distance")
        suffix = f" (distance: {float(distance):.4f})" if isinstance(distance, (float, int)) else ""
        console.print(f"[{idx}] {chunk.content}{suffix}")



def display_query_info(query_output: RAGQueryAgentOutputSchema) -> None:
    """Display information about the generated query."""
    console.print("\n[bold yellow]Semantic Search Strategy[/bold yellow]")
    console.print(f"Query: {query_output.query}")
    console.print(f"Reasoning: {query_output.reasoning}")



def display_answer(qa_output: RAGQuestionAnsweringAgentOutputSchema) -> None:
    """Display the reasoning and answer from the QA agent."""
    console.print("\n[bold green]Reasoning[/bold green]")
    console.print(qa_output.reasoning)
    console.print("\n[bold blue]Answer[/bold blue]")
    console.print(qa_output.answer)


def display_transcription(stt_output: SpeechToTextOutputSchema) -> None:
    """Display transcription result from the STT agent."""
    console.print("\n[bold green]Transcription[/bold green]")
    console.print(stt_output.transcription)


def display_tasking(task_output: TaskingOutputSchema) -> None:
    """Display structured command extracted by the tasking agent."""
    console.print("\n[bold blue]Structured Command[/bold blue]")
    console.print(f"Asset: {task_output.asset}")
    console.print(f"Tasking Verb: {task_output.tasking_verb}")
    console.print(f"Target: {task_output.target}")
    if task_output.classification:
        console.print(f"Classification: {task_output.classification}")
    if task_output.time_constraint:
        console.print(f"Time Constraint: {task_output.time_constraint}")



def chat_loop(environment: OrchestrationEnvironment) -> None:
    """Main chat loop driven by the orchestration agent."""
    display_welcome()

    while True:
        try:
            user_message = console.input("\n[bold blue]Your request:[/bold blue] ").strip()

            if not user_message:
                console.print("[dim]Please enter a request or type 'exit'.[/dim]")
                continue

            if user_message.lower() in ChatConfig.exit_commands:
                console.print("Goodbye! Thanks for using the Orchestrated Chatbot.")
                break

            console.print("\n" + "─" * 40)
            console.print("[bold magenta]Routing your request...[/bold magenta]")

            orchestration_response: OrchestratorOutputSchema = orchestrator_agent.run(
                OrchestratorInputSchema(chat_message=user_message)
            )

            tool_choice = orchestration_response.tool
            console.print(f"[dim]Tool selected: {tool_choice}[/dim]")

            if tool_choice == "stt":
                stt_params = orchestration_response.tool_parameters
                if not isinstance(stt_params, SpeechToTextInputSchema):
                    stt_params = SpeechToTextInputSchema(**stt_params.model_dump())

                stt_output = environment.stt_agent.execute(stt_params)
                display_transcription(stt_output)

            elif tool_choice == "tasking":
                task_params = orchestration_response.tool_parameters
                if not isinstance(task_params, TaskingInputSchema):
                    task_params = TaskingInputSchema(**task_params.model_dump())

                task_output = tasking_agent.run(task_params)
                display_tasking(task_output)

            elif tool_choice == "qa":
                qa_params = orchestration_response.tool_parameters
                if not isinstance(qa_params, RAGQuestionAnsweringAgentInputSchema):
                    qa_params = RAGQuestionAnsweringAgentInputSchema(**qa_params.model_dump())

                question = qa_params.question

                query_output = query_agent.run(RAGQueryAgentInputSchema(user_message=question))
                display_query_info(query_output)

                search_results = environment.chroma_db.query(
                    query_text=query_output.query,
                    n_results=NUM_CHUNKS_TO_RETRIEVE,
                )

                environment.rag_context.chunks = [
                    ChunkItem(content=doc, metadata={"chunk_id": doc_id, "distance": distance})
                    for doc, doc_id, distance in zip(
                        search_results["documents"],
                        search_results["ids"],
                        search_results["distances"],
                    )
                ]

                display_chunks(environment.rag_context.chunks)

                qa_output = qa_agent.run(RAGQuestionAnsweringAgentInputSchema(question=question))
                display_answer(qa_output)

            else:  # pragma: no cover - defensive branch
                console.print(f"[bold red]Unknown tool selected: {tool_choice}[/bold red]")

            console.print("\n" + "─" * 40)

        except Exception as exc:  # pragma: no cover - console app flow
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            console.print("[dim]Please try again or type 'exit' to quit.[/dim]")
