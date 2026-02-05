"""Schema creator — generates JSON Schema + system prompt from natural language."""

from typing import Any, Callable, Dict, Optional, Union

from jig.client import LMStudioClient
from jig.ollama_client import OllamaClient
from jig.constants import CREATOR_TEMPERATURE
from jig.repository import PairingRepository
from jig.utils import normalize_schema_for_backend, sanitize_filename


CREATOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_name": {
            "type": "string",
            "description": "CamelCase identifier for this schema",
            "minLength": 1,
        },
        "description": {
            "type": "string",
            "description": "One-line description of what this schema does",
            "minLength": 1,
        },
        "system_prompt": {
            "type": "string",
            "description": "Detailed system prompt",
            "minLength": 1,
        },
        "response_schema": {
            "type": "object",
            "description": "Valid JSON Schema (draft-07) with strict typing",
        },
    },
    "required": ["schema_name", "description", "system_prompt", "response_schema"],
}

CREATOR_SYSTEM_PROMPT = (
    "You are an expert JSON Schema architect. Create:\n"
    "1) A precise system prompt for the AI task\n"
    "2) A strict JSON Schema enforcing the output structure\n\n"
    "Rules for response_schema:\n"
    "- Use strict typing: additionalProperties: false\n"
    "- All properties must be required\n"
    "- Include descriptions for each field\n"
    "- Use JSON Schema Draft 7"
)


class SchemaCreator:
    """Creates schema+prompt pairings via LLM generation."""

    def __init__(
        self,
        client: Union[LMStudioClient, OllamaClient],
        repository: Optional[PairingRepository] = None,
        confirm_overwrite: Optional[Callable[[str], bool]] = None,
    ):
        self.client = client
        self.repo = repository or PairingRepository()
        self.confirm_overwrite = confirm_overwrite

    def create(
        self,
        purpose: str,
        name: str = "generated",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate and save a new schema+prompt pairing.

        Args:
            purpose: Natural language description of the AI task
            name: Pairing name (used as directory and schema id)
            force: If True, overwrite without confirmation

        Returns:
            Generated result dict with schema_name, description, system_prompt, response_schema.
            Empty dict if user declined overwrite.
        """
        safe_name = sanitize_filename(name)
        target_dir = self.repo.path(safe_name)

        if target_dir.exists() and not force:
            if self.confirm_overwrite is None:
                return {}  # No confirm callback: skip overwrite for safety
            if not self.confirm_overwrite(
                f"Pairing '{safe_name}' already exists. Overwrite?"
            ):
                return {}

        messages = [
            {"role": "system", "content": CREATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Create schema for: {purpose}"},
        ]
        creator_schema, _ = normalize_schema_for_backend(CREATOR_SCHEMA)

        result = self.client.generate_structured(
            messages,
            creator_schema,
            temperature=CREATOR_TEMPERATURE,
        )

        # Use user-provided name instead of LLM-generated schema_name
        result["schema_name"] = safe_name

        response_schema = result.get("response_schema")
        if not response_schema:
            raise RuntimeError("Generated schema missing response_schema")

        self.repo.save(
            name=safe_name,
            schema=response_schema,
            prompt=result["system_prompt"],
            meta={
                "name": safe_name,
                "description": result.get("description", ""),
                "model": self.client.model,
            },
        )

        return result
