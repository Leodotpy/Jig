"""LM Studio API client — OpenAI-compatible chat completions with JSON Schema."""

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from requests.exceptions import RequestException
from openai import OpenAI

from jig.constants import API_KEY


class LMStudioClient:
    """
    Client for LM Studio's OpenAI-compatible API.

    Supports structured output via response_format with json_schema.
    """

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        timeout_s: float = 3.0,
        api_key: str = API_KEY,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.model = model
        self._client = OpenAI(base_url=self.base_url, api_key=api_key)

    def _get_models(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/models"
        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        return models if isinstance(models, list) else []

    def preflight(self) -> Tuple[bool, str]:
        """Check connectivity and list available models."""
        try:
            models = self._get_models()
            if not models:
                return False, f"Reached {self.base_url} but no models loaded"
            names = [m.get("id", "unknown") for m in models]
            return True, f"Connected | Models: {', '.join(names[:3])}"
        except RequestException as e:
            return False, f"Connection failed: {e}"
        except Exception as e:
            return False, f"Error: {e}"

    def list_models(self) -> List[str]:
        """Return list of available model IDs for selection."""
        models = self._get_models()
        return [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]

    def ensure_model(self) -> str:
        """Resolve and validate model ID. Raises RuntimeError if unavailable."""
        models = self._get_models()
        ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]

        if not ids:
            raise RuntimeError(
                f"No models found at {self.base_url}/models. Load a model in LM Studio first."
            )

        if self.model:
            if self.model in ids:
                return self.model
            partial = [mid for mid in ids if self.model.lower() in (mid or "").lower()]
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
            messages: Chat messages (OpenAI format; content may be str or list of parts)
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
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": schema,
            },
        }

        if images:
            messages = list(messages)
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    text = messages[i].get("content", "")
                    if isinstance(text, list):
                        text = next(
                            (
                                p.get("text", "")
                                for p in text
                                if p.get("type") == "text"
                            ),
                            "",
                        )
                    parts: List[Dict[str, Any]] = [{"type": "text", "text": text or ""}]
                    for b64, mime in images:
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            }
                        )
                    messages[i] = {**messages[i], "content": parts}
                    break

        try:
            resp = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("Empty response from model")
            return json.loads(content)
        except json.JSONDecodeError as e:
            raw = (content or "")[:500]
            raise RuntimeError(
                f"Invalid JSON returned. Parse error: {e}. Content preview: {raw}..."
            )
        except Exception as e:
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
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": schema,
            },
        }
        if images:
            messages = list(messages)
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    text = messages[i].get("content", "")
                    if isinstance(text, list):
                        text = next(
                            (
                                p.get("text", "")
                                for p in text
                                if p.get("type") == "text"
                            ),
                            "",
                        )
                    parts: List[Dict[str, Any]] = [{"type": "text", "text": text or ""}]
                    for b64, mime in images:
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            }
                        )
                    messages[i] = {**messages[i], "content": parts}
                    break
        try:
            stream = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            content = ""
            for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                content += delta
                yield content, None
            result = json.loads(content) if content.strip() else {}
            yield content, result
        except Exception as e:
            raise RuntimeError(f"Inference error: {e}")
