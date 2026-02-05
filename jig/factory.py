"""Client factory — connection handling and backend selection."""

from typing import Literal, Optional, Union

from jig.client import LMStudioClient
from jig.constants import (
    COMMON_PORTS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    OLLAMA_DEFAULT_PORT,
)
from jig.ollama_client import OllamaClient

BackendKind = Literal["lmstudio", "ollama", "auto"]
LLMClient = Union[LMStudioClient, OllamaClient]


def build_lmstudio_url(host: str, port: int) -> str:
    """Build LM Studio API base URL (OpenAI-compatible)."""
    return f"http://{host}:{port}/v1"


def build_ollama_url(host: str, port: int) -> str:
    """Build Ollama API base URL."""
    return f"http://{host}:{port}"


def create_client(
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    model: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    auto_probe: bool = False,
    backend: BackendKind = "auto",
) -> LLMClient:
    """
    Create an LLM client for the given backend.

    Args:
        host: Server host
        port: Server port (default depends on backend: 1234 for LM Studio, 11434 for Ollama)
        model: Model name/ID (optional, uses first available if not specified)
        timeout: Connection timeout in seconds
        auto_probe: If connection fails, try common ports (LM Studio only)
        backend: "lmstudio" | "ollama" | "auto" (try LM Studio then Ollama)

    Returns:
        LMStudioClient or OllamaClient

    Raises:
        RuntimeError: If connection fails
    """
    if backend == "ollama":
        return _create_ollama_client(host, port or OLLAMA_DEFAULT_PORT, model, timeout)
    if backend == "lmstudio":
        return _create_lmstudio_client(
            host, port or DEFAULT_PORT, model, timeout, auto_probe
        )
    # auto: try LM Studio first, then Ollama
    try:
        return _create_lmstudio_client(
            host, port or DEFAULT_PORT, model, timeout, auto_probe
        )
    except RuntimeError:
        return _create_ollama_client(host, port or OLLAMA_DEFAULT_PORT, model, timeout)


def _create_lmstudio_client(
    host: str,
    port: int,
    model: Optional[str],
    timeout: float,
    auto_probe: bool,
) -> LMStudioClient:
    """Create LM Studio client with optional port auto-detection."""
    base_url = build_lmstudio_url(host, port)
    client = LMStudioClient(base_url=base_url, model=model, timeout_s=timeout)
    ok, _ = client.preflight()

    if ok:
        return client

    if not auto_probe:
        raise RuntimeError(
            "Cannot connect to LM Studio. Ensure the server is running (Developer tab), "
            f"check port {port}, or use --auto-probe to scan common ports."
        )

    for p in COMMON_PORTS:
        if p == port:
            continue
        candidate = LMStudioClient(
            base_url=build_lmstudio_url(host, p),
            model=model,
            timeout_s=timeout,
        )
        ok2, _ = candidate.preflight()
        if ok2:
            return candidate

    raise RuntimeError("Connection failed on all common ports (LM Studio)")


def _create_ollama_client(
    host: str,
    port: int,
    model: Optional[str],
    timeout: float,
) -> OllamaClient:
    """Create Ollama client."""
    base_url = build_ollama_url(host, port)
    client = OllamaClient(base_url=base_url, model=model, timeout_s=timeout)
    ok, msg = client.preflight()
    if not ok:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. "
            "Ensure Ollama is running (e.g. ollama serve). "
            f"Details: {msg}"
        )
    return client
