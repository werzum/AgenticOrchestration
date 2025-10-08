# Agentic Orchestration Chatbot

This project provides an orchestrated assistant that can switch between speech transcription, structured task extraction, and retrieval-augmented QA. An orchestration agent evaluates each user turn and routes the request to the most relevant tool, producing a single seamless chat experience.

## Features
- **Tool orchestration** – The `orchestrator_agent` decides between `stt`, `tasking`, and `qa` tools based on natural-language instructions.
- **Retrieval-Augmented Generation** – The QA path builds semantic queries, searches a ChromaDB vector store, and synthesizes answers with source context.
- **Speech transcription** – Whisper-based speech-to-text is available through the orchestrated chat or the standalone transcription loop.
- **Task structuring** – Free-form operational commands can be normalised into a structured schema (asset, verb, target, etc.).

## Requirements
- Python 3.12+
- [Ollama](https://ollama.com/) with models used in the agents (e.g. `ollama run gemma3`, `ollama run llama3.2`)
- Whisper dependencies (PyTorch, CUDA) if you plan to run the speech-to-text tool
- Python libraries: `rich`, `chromadb`, `instructor`, `atomic-agents`

Create a virtual environment and install dependencies, for example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install rich chromadb instructor atomic-agents
```

## Running the orchestrated chat

Prepare a folder of Markdown documents that should populate the knowledge base. By default the app uses `~/Documents/KnowledgeBase`, but you can point it elsewhere with `--documents`:

```bash
python3 -m rag_chatbot.main --mode chat --documents /path/to/markdown
```

The chat loop supports natural-language requests such as:
- "Please transcribe the audio file at ~/recordings/checklist.mp3" → routes to the STT tool.
- "Parse this instruction: hold flight 482 at gate A12" → routes to the tasking tool.
- "How do we deploy the ground crew automation service?" → routes to the RAG QA pipeline.

Type `exit` (or `/exit`, `/quit`) to leave the session.

## Optional: transcription utility

A standalone transcription loop remains available if you want a focussed audio-to-text workflow:

```bash
python3 -m rag_chatbot.main --mode transcribe
```

This mode still chains the tasking agent after transcription to provide structured outputs.

## Project layout

```
rag_chatbot/
├── agents/            # Tool agents, including the orchestrator
├── app/               # Console chat loop and helpers
├── context/           # Context provider utilities
├── services/          # ChromaDB and Whisper integrations
└── main.py            # CLI entry point
```

Vector store data is persisted under `chroma_db/` by default. You can safely delete the directory to rebuild embeddings from scratch.

## Customisation tips
- Update `rag_chatbot/config.py` to change model aliases, document paths, and retrieval parameters.
- Adjust prompts for individual agents inside `rag_chatbot/agents/` to tailor reasoning or output format.
- Extend `orchestrator_agent` if you want to introduce new tools or routing criteria.

Happy hacking!
