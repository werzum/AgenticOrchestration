"""Agent responsible for structuring tasking commands from transcribed audio."""

from __future__ import annotations

from typing import Optional

import instructor
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from pydantic import Field

from rag_chatbot.config import ChatConfig


class TaskingInputSchema(BaseIOSchema):
    """Input schema for the tasking agent."""

    input_string: str = Field(..., description="String of the transcribed voice message")


class TaskingOutputSchema(BaseIOSchema):
    """Output schema for the tasking agent."""

    asset: str = Field(..., description="The resource handling the instruction (e.g., 'ground crew', 'tower')")
    tasking_verb: str = Field(..., description="The operational action to perform (e.g., 'pushback', 'hold')")
    target: str = Field(..., description="The location or flight identifier involved (e.g., 'gate A12', 'flight 482')")
    classification: Optional[str] = Field(None, description="Operational status (e.g., 'ready', 'delayed')")
    time_constraint: Optional[str] = Field(None, description="Time reference if mentioned (e.g., 'immediately', 'within 5 minutes')")


tasking_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_provider("ollama/gemma3"),
        model=ChatConfig.model,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You assist civil aviation operations by structuring instructions from radio transcripts.",
                "Focus on airport ground coordination, flight handling, and passenger service workflows.",
            ],
            steps=[
                "Receive the transcript.",
                "1. Clean the transcription by removing clear repetitions or speech disfluencies while keeping intent intact.",
                "2. Identify these components in the message:\n- Asset: ground resource or control entity (e.g., 'ground crew', 'tower').\n- Tasking Verb: operational action (e.g., 'pushback', 'taxi', 'hold', 'service', 'assist').\n- Target: flight identifier or location (e.g., 'flight 482', 'runway 27', 'gate B4').\n- Classification: operational status (e.g., 'ready', 'delayed', 'boarding', 'arrived').\n- Time Constraint: time-related phrasing (e.g., 'immediately', 'within 10 minutes', 'after arrival').",
                "3. Use the following standardized vocabulary where possible, selecting the closest match if needed:\n- Asset: 'ground crew', 'cabin crew', 'maintenance', 'tower', 'operations'.\n- TaskingVerb: 'pushback', 'taxi', 'hold', 'service', 'assist', 'dispatch'.\n- ClassificationWords: 'ready', 'delayed', 'boarding', 'deplaning', 'arrived'.\n- TimeConstraints: 'immediately', 'within 5 minutes', 'within 10 minutes', 'after arrival', 'before departure'.",
                "4. Return a structured JSON response that follows the expected schema without extra commentary.",
            ],
            output_instructions=["Provide the result strictly in the expected schema."],
        ),
        input_schema=TaskingInputSchema,
        output_schema=TaskingOutputSchema,
    )
)
