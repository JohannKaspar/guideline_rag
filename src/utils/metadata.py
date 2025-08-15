"""Metadata processing utilities."""

import re
from enum import Enum
from typing import Any


def flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metadata dictionaries."""
    flat_meta = {}

    def _flatten(prefix: str, value: Any):
        if isinstance(value, dict):
            for k, v in value.items():
                _flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(value, list | tuple):
            for i, item in enumerate(value):
                _flatten(f"{prefix}[{i}]", item)
        elif isinstance(value, Enum):
            flat_meta[prefix] = value.value
        elif value is None:
            pass
        elif "hash" in prefix.lower():
            flat_meta[prefix] = str(value)
        else:
            flat_meta[prefix] = value

    _flatten("", metadata)
    return flat_meta


def normalize_whitespace(text: str) -> str:
    """Reduce multiple spaces/tabs to a single space."""
    return re.sub(r"[ \t]+", " ", text)


def clean_metadata_for_storage(metadata: dict[str, Any]) -> dict[str, Any]:
    """Clean metadata by removing base64 images and other large data."""
    return {k: v for k, v in metadata.items() if "image" not in k.lower() and not isinstance(v, bytes)}
