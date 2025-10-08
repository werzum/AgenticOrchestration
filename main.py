"""Convenience entrypoint that proxies to the package CLI."""

from rag_chatbot.main import main


if __name__ == "__main__":  # pragma: no cover - CLI trampoline
    main()
