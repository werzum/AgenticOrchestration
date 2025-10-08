"""Agent responsible for tool selection across the orchestration workflow."""

from __future__ import annotations

from typing import Literal, Union

import instructor
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from pydantic import Field

from rag_chatbot.agents.qa_agent import RAGQuestionAnsweringAgentInputSchema
from rag_chatbot.agents.stt_agent import SpeechToTextInputSchema
from rag_chatbot.agents.tasking_agent import TaskingInputSchema
from rag_chatbot.config import ChatConfig


class OrchestratorInputSchema(BaseIOSchema):
    """User message routed through the orchestrator."""

    chat_message: str = Field(..., description="The latest user instruction or request.")


class OrchestratorOutputSchema(BaseIOSchema):
    """Tool decision emitted by the orchestrator."""

    tool: Literal["stt", "tasking", "qa"] = Field(..., description="Selected tool identifier.")
    tool_parameters: Union[
        SpeechToTextInputSchema,
        TaskingInputSchema,
        RAGQuestionAnsweringAgentInputSchema,
    ] = Field(..., description="Input payload for the chosen tool.")


orchestrator_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_provider("ollama/gemma3"),
        model=ChatConfig.model,
        input_schema=OrchestratorInputSchema,
        output_schema=OrchestratorOutputSchema,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are the orchestration layer for a multimodal assistant.",
                "Available tools:\n- stt: transcribe audio files using Whisper.\n- tasking: structure free-form operational commands.\n- qa: answer project questions using a knowledge base.",
            ],
            steps=[
                "Inspect the user's latest message.",
                "Decide whether they want audio transcription, task structuring, or knowledge-base QA.",
                "Prepare the tool-specific input payload.",
            ],
            output_instructions=[
                "Always return a JSON object that matches the output schema.",
                "Tool-specific guidance:\n- stt: expect an `audio_file_path`.\n- tasking: expect an `input_string`.",
                "- qa: expect a `question` to forward to the retrieval pipeline.",
                "If none of the tools apply, default to `qa` and pass the original message as `question`.",
            ],
        ),
    )
)

__all__ = [
    "OrchestratorInputSchema",
    "OrchestratorOutputSchema",
    "orchestrator_agent",
]

