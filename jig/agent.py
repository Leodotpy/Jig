"""Schema agent — runs structured inference using a pairing."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from jig.client import LMStudioClient
from jig.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from jig.ollama_client import OllamaClient
from jig.utils import (
    load_images_as_base64,
    normalize_schema_for_backend,
    read_input_data,
    restore_response_keys,
    resolve_pairing_path,
)


class SchemaAgent:
    """Runs structured inference using a schema+prompt pairing."""

    def __init__(self, client: Union[LMStudioClient, OllamaClient]):
        self.client = client

    def run(
        self,
        input_data: str,
        schema_ref: str,
        prompt_ref: Optional[str] = None,
        output_path: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference with the given pairing.

        Args:
            input_data: User input text or path to .txt file
            schema_ref: Schema file path or pairing name
            prompt_ref: Prompt file path or pairing name (defaults to schema_ref)
            output_path: Optional path to save JSON result
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            image_paths: Optional list of image file paths for vision models

        Returns:
            Parsed JSON result from the model
        """
        input_data = read_input_data(input_data)
        prompt_ref = prompt_ref or schema_ref

        schema_path = resolve_pairing_path(schema_ref, "schema")
        prompt_path = resolve_pairing_path(prompt_ref, "prompt")

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        prompt = prompt_path.read_text(encoding="utf-8")
        schema, rename_map = normalize_schema_for_backend(schema)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_data or "(no text)"},
        ]

        images_data: Optional[List[tuple]] = None
        if image_paths:
            images_data = load_images_as_base64(image_paths)
            if not images_data:
                raise ValueError(
                    "No valid images could be loaded from the provided paths. "
                    "Use vision-capable models (e.g. zai-org/GLM-4.6, llava, deepseek-vl)."
                )

        result = self.client.generate_structured(
            messages,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            images=images_data,
        )
        result = restore_response_keys(result, rename_map)

        if output_path:
            Path(output_path).write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return result

    def run_stream(
        self,
        input_data: str,
        schema_ref: str,
        prompt_ref: Optional[str] = None,
        output_path: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        image_paths: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, Optional[Dict[str, Any]]]]:
        """Stream inference: yields (accumulated_text, None) then (full_text, parsed_result)."""
        input_data = read_input_data(input_data)
        prompt_ref = prompt_ref or schema_ref
        schema_path = resolve_pairing_path(schema_ref, "schema")
        prompt_path = resolve_pairing_path(prompt_ref, "prompt")
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        prompt = prompt_path.read_text(encoding="utf-8")
        schema, rename_map = normalize_schema_for_backend(schema)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_data or "(no text)"},
        ]
        images_data: Optional[List[tuple]] = None
        if image_paths:
            images_data = load_images_as_base64(image_paths)
            if not images_data:
                raise ValueError(
                    "No valid images could be loaded. Use vision-capable models "
                    "(e.g. zai-org/GLM-4.6, llava, deepseek-vl)."
                )
        for content, result in self.client.generate_structured_stream(
            messages,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            images=images_data,
        ):
            if result is not None:
                result = restore_response_keys(result, rename_map)
                if output_path:
                    Path(output_path).write_text(
                        json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                yield content, result
            else:
                yield content, None
