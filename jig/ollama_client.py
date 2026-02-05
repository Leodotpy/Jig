"""Ollama API client — native chat API with JSON Schema structured output."""

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from requests.exceptions import RequestException


class OllamaClient:
    """
    Client for Ollama's native API.

    Supports structured output via the `format` parameter (JSON Schema).
    """

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        timeout_s: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.model = model

    def _get_models(self) -> List[Dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        return models if isinstance(models, list) else []

    def preflight(self) -> Tuple[bool, str]:
        """Check connectivity and list available models."""
        try:
            models = self._get_models()
            if not models:
                return False, f"Reached {self.base_url} but no models found"
            names = [m.get("name", m.get("model", "unknown")) for m in models]
            return True, f"Connected | Models: {', '.join(names[:3])}"
        except RequestException as e:
            return False, f"Connection failed: {e}"
        except Exception as e:
            return False, f"Error: {e}"

    def list_models(self) -> List[str]:
        """Return list of available model names for selection."""
        models = self._get_models()
        ids = []
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                ids.append(name)
        return ids

    def ensure_model(self) -> str:
        """Resolve and validate model name. Raises RuntimeError if unavailable."""
        models = self._get_models()
        ids = []
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                ids.append(name)

        if not ids:
            raise RuntimeError(
                f"No models found at {self.base_url}/api/tags. "
                "Pull a model first: ollama pull zai-org/GLM-4.6"
            )

        if self.model:
            if self.model in ids:
                return self.model
            short = self.model.split(":")[0].lower()
            partial = [
                mid
                for mid in ids
                if short in mid.lower() or mid.lower().startswith(short)
            ]
            if len(partial) == 1:
                self.model = partial[0]
                return self.model
            raise RuntimeError(
                f"Model '{self.model}' not found. Available: {', '.join(ids)}"
            )

        self.model = ids[0]
        return self.model

    def generate_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        images: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate completion with strict JSON Schema output.

        Args:
            messages: Chat messages (role, content)
            schema: JSON Schema for the response
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            images: Optional list of (base64_string, mime_type) for vision models

        Returns:
            Parsed JSON response as dict

        Raises:
            RuntimeError: On empty response, invalid JSON, or API error
        """
        model_id = self.ensure_model()
        if images:
            messages = [dict(m) for m in messages]
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i]["images"] = [b64 for b64, _ in images]
                    break
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_s * 10,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("message") or {}).get("content")
            if content is None or content == "":
                raise RuntimeError("Empty response from model")
            return json.loads(content)
        except json.JSONDecodeError as e:
            raw = (content or "")[:500]
            raise RuntimeError(
                f"Invalid JSON returned. Parse error: {e}. Content preview: {raw}..."
            )
        except RequestException as e:
            raise RuntimeError(f"Inference error: {e}")

    def generate_structured_stream(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        images: Optional[List[Tuple[str, str]]] = None,
    ) -> Iterator[Tuple[str, Optional[Dict[str, Any]]]]:
        """Stream structured output: yields (accumulated_text, None) then (full_text, parsed_result)."""
        model_id = self.ensure_model()
        if images:
            messages = [dict(m) for m in messages]
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i]["images"] = [b64 for b64, _ in images]
                    break
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "format": schema,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        content = ""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout_s * 10,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message") or {}
                part = msg.get("content") or ""
                content += part
                if data.get("done"):
                    result = json.loads(content) if content.strip() else {}
                    yield content, result
                    return
                yield content, None
            result = json.loads(content) if content.strip() else {}
            yield content, result
        except RequestException as e:
            raise RuntimeError(f"Inference error: {e}")
