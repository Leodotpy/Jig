"""Pairing repository — CRUD for schema/prompt pairings on disk."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from jig.constants import PAIRINGS_DIR
from jig.utils import load_json_safe, sanitize_filename


class PairingRepository:
    """
    Repository for schema+prompt pairings.

    Each pairing lives in pairings/<name>/ with schema.json, prompt.txt, meta.json.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or PAIRINGS_DIR

    def path(self, name: str) -> Path:
        """Directory path for a pairing by name."""
        return self.base_dir / sanitize_filename(name)

    def exists(self, name: str) -> bool:
        """Check if pairing directory exists."""
        return self.path(name).exists()

    def is_complete(self, name: str) -> bool:
        """Check if pairing has both schema.json and prompt.txt."""
        p = self.path(name)
        return (p / "schema.json").exists() and (p / "prompt.txt").exists()

    def list_all(self) -> Iterator[Dict[str, Any]]:
        """List all pairings with metadata. Skips backup directories (*_backup_*)."""
        if not self.base_dir.exists():
            return
        for d in sorted(self.base_dir.iterdir()):
            if not d.is_dir():
                continue
            if "_backup_" in d.name:
                continue
            meta = load_json_safe(d / "meta.json")
            yield {
                "name": d.name,
                "description": meta.get("description", "")[:40],
                "schema_exists": (d / "schema.json").exists(),
                "prompt_exists": (d / "prompt.txt").exists(),
            }

    def load(self, name: str) -> Dict[str, Any]:
        """Load schema, prompt, and meta for a pairing. Raises if missing."""
        p = self.path(name)
        schema_path = p / "schema.json"
        prompt_path = p / "prompt.txt"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        schema = load_json_safe(schema_path)
        if not schema:
            raise ValueError(f"Invalid or empty schema: {schema_path}")
        prompt = prompt_path.read_text(encoding="utf-8")
        meta = load_json_safe(p / "meta.json")
        return {"schema": schema, "prompt": prompt, "meta": meta}

    def save(
        self,
        name: str,
        schema: Dict[str, Any],
        prompt: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save pairing. Creates backup if directory exists and has content."""
        target = self.path(name)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if target.exists() and any(target.iterdir()):
            backup_name = (
                f"{target.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            backup_dir = target.parent / backup_name
            target.rename(backup_dir)
            target = self.base_dir / target.name

        target.mkdir(parents=True, exist_ok=True)

        (target / "schema.json").write_text(
            json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (target / "prompt.txt").write_text(prompt, encoding="utf-8")

        meta_data = meta or {}
        meta_data.setdefault("name", name)
        meta_data.setdefault("created", datetime.now().isoformat())
        (target / "meta.json").write_text(
            json.dumps(meta_data, indent=2), encoding="utf-8"
        )

        return target
