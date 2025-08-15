"""Utilities package."""

from .config import Config
from .metadata import clean_metadata_for_storage, flatten_metadata, normalize_whitespace

# Import optional modules with error handling
try:
    from .embeddings import HuggingFaceEmbeddings, get_embeddings_model

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    HuggingFaceEmbeddings = None
    get_embeddings_model = None

try:
    from .clients import get_chat_model, get_genai_hub_client, get_openai_embeddings

    CLIENTS_AVAILABLE = True
except ImportError:
    CLIENTS_AVAILABLE = False
    get_genai_hub_client = None
    get_chat_model = None
    get_openai_embeddings = None

__all__ = [
    "Config",
    "flatten_metadata",
    "normalize_whitespace",
    "clean_metadata_for_storage",
    "HuggingFaceEmbeddings",
    "get_embeddings_model",
    "get_genai_hub_client",
    "get_chat_model",
    "get_openai_embeddings",
    "EMBEDDINGS_AVAILABLE",
    "CLIENTS_AVAILABLE",
]
