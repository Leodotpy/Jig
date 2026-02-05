"""
Jig — Structured output pipeline for local and multimodal LLMs.

Create JSON Schema + prompt pairings and run structured inference against
LM Studio (OpenAI-compatible API) or Ollama (native API).
"""

__version__ = "1.01"

from jig.agent import SchemaAgent
from jig.client import LMStudioClient
from jig.creator import SchemaCreator
from jig.factory import create_client
from jig.ollama_client import OllamaClient
from jig.repository import PairingRepository


def get_pairing(name: str) -> dict:
    """
    Load a pairing by name from the default pairings directory.

    Returns:
        Dict with keys: "schema" (JSON Schema dict), "prompt" (str), "meta" (dict).

    Raises:
        FileNotFoundError: If schema or prompt file is missing.
        ValueError: If schema JSON is invalid.
    """
    return PairingRepository().load(name)


__all__ = [
    "__version__",
    "LMStudioClient",
    "OllamaClient",
    "SchemaCreator",
    "SchemaAgent",
    "PairingRepository",
    "create_client",
    "get_pairing",
]
