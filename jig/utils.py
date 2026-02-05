"""Path resolution, sanitization, image loading, schema normalization, and I/O helpers."""

import base64
import copy
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from jig.constants import PAIRINGS_DIR

# MIME types for common image extensions
IMAGE_EXT_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def sanitize_filename(name: str) -> str:
    """Convert arbitrary string to safe directory name."""
    safe = re.sub(r"[^\w\s-]", "", name).strip().lower()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe or "untitled"


def resolve_pairing_path(name_or_path: str, ext: str = "json") -> Path:
    """
    Resolve pairing reference to actual file path.

    - If name_or_path contains path separators or ends with .json/.txt, treat as direct path
    - Else resolve to pairings/{name}/schema.json or prompt.txt
    """
    if (
        "/" in name_or_path
        or "\\" in name_or_path
        or name_or_path.endswith(".json")
        or name_or_path.endswith(".txt")
    ):
        return Path(name_or_path)

    safe_name = sanitize_filename(name_or_path)
    base = PAIRINGS_DIR / safe_name

    if ext in ("json", "schema"):
        return base / "schema.json"
    if ext in ("txt", "prompt"):
        return base / "prompt.txt"
    return base


def read_input_data(input_data: str) -> str:
    """
    Read input from string or file.
    If input_data is a path to an existing .txt file, read it. Otherwise return as-is.
    """
    path = Path(input_data)
    if path.exists() and path.suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return input_data
    return input_data


def load_json_safe(path: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict on error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# Property names that conflict with JSON Schema keywords (some backends reject them inside "properties")
_RESERVED_PROP_NAMES = frozenset(
    {"required", "properties", "type", "items", "additionalProperties"}
)
_RESERVED_RENAME_PREFIX = "_prop_"


def normalize_schema_for_backend(
    schema: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Normalize a JSON Schema so backends (LM Studio, Ollama) accept it.

    - Ensures "required" at every object level is an array of strings (never boolean/object).
    - Renames any property named "required", "properties", "type", etc. to "_prop_required", etc.,
      so backends do not misparse them. Returns (normalized_schema, rename_map) where rename_map
      maps new_name -> original_name for use when restoring the response.
    """
    rename_map: Dict[str, str] = {}

    def _norm(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        out = copy.deepcopy(obj)
        props = out.get("properties")
        if isinstance(props, dict):
            new_props: Dict[str, Any] = {}
            for k, v in props.items():
                if k in _RESERVED_PROP_NAMES:
                    new_name = _RESERVED_RENAME_PREFIX + k
                    rename_map[new_name] = k
                    new_props[new_name] = _norm(v)
                else:
                    new_props[k] = _norm(v)
            out["properties"] = new_props
        items = out.get("items")
        if isinstance(items, dict):
            out["items"] = _norm(items)
        elif isinstance(items, list):
            out["items"] = [_norm(x) if isinstance(x, dict) else x for x in items]
        req = out.get("required")
        if req is not None:
            if isinstance(req, list) and all(isinstance(x, str) for x in req):
                pass
            elif isinstance(req, bool):
                out["required"] = list(out.get("properties", {}).keys()) if req else []
            elif isinstance(req, dict):
                out["required"] = [k for k, v in req.items() if v]
            else:
                out["required"] = (
                    list(out.get("properties", {}).keys())
                    if out.get("properties")
                    else []
                )
        elif isinstance(out.get("properties"), dict) and out["properties"]:
            out["required"] = []
        for key in ("anyOf", "oneOf", "allOf"):
            if key in out and isinstance(out[key], list):
                out[key] = [_norm(x) if isinstance(x, dict) else x for x in out[key]]
        if "$defs" in out and isinstance(out["$defs"], dict):
            out["$defs"] = {
                k: _norm(v) if isinstance(v, dict) else v
                for k, v in out["$defs"].items()
            }
        return out

    if not schema or not isinstance(schema, dict):
        return schema, {}
    normalized = _norm(schema)

    # Fix required array: use original names for required list (we didn't rename in required, we need to use schema names)
    # Actually we renamed props so required list should reference the NEW names in the normalized schema
    def _fix_required(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        props = obj.get("properties")
        if isinstance(props, dict):
            req = obj.get("required")
            if isinstance(req, list):
                new_req = []
                for r in req:
                    if r in _RESERVED_PROP_NAMES:
                        new_req.append(_RESERVED_RENAME_PREFIX + r)
                    else:
                        new_req.append(r)
                obj["required"] = new_req
        for v in (obj.get("properties") or {}).values():
            _fix_required(v)
        if "items" in obj and isinstance(obj["items"], dict):
            _fix_required(obj["items"])

    _fix_required(normalized)
    return normalized, rename_map


def restore_response_keys(data: Any, rename_map: Dict[str, str]) -> Any:
    """Recursively restore keys in API response that were renamed by normalize_schema_for_backend."""
    if not rename_map:
        return data
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            key = rename_map.get(k, k)
            out[key] = restore_response_keys(v, rename_map)
        return out
    if isinstance(data, list):
        return [restore_response_keys(x, rename_map) for x in data]
    return data


def _mime_for_path(path: Path) -> str:
    """Return MIME type for path; default image/jpeg."""
    return IMAGE_EXT_MIME.get(path.suffix.lower(), "image/jpeg")


def load_images_as_base64(paths: List[str]) -> List[Tuple[str, str]]:
    """
    Load image(s) from file path(s) or data URL(s) into (base64, mime) pairs.

    Each path can be:
    - A file path to an image (e.g. photo.jpg)
    - A data URL: data:image/jpeg;base64,...

    Returns:
        List of (base64_string, mime_type) for each image.
    """
    result: List[Tuple[str, str]] = []
    for p in paths:
        if not p or not p.strip():
            continue
        s = p.strip()
        if s.startswith("data:") and ";base64," in s:
            # data:image/jpeg;base64,XXXX
            header, b64 = s.split(";base64,", 1)
            mime = header.replace("data:", "", 1).strip() or "image/jpeg"
            result.append((b64.strip(), mime))
            continue
        path = Path(s)
        if not path.exists():
            continue
        mime = _mime_for_path(path)
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            result.append((b64, mime))
        except OSError:
            continue
    return result
