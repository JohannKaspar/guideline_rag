"""Core modules package."""

# Import core modules with error handling
try:
    from .vector_store import VectorStoreManager

    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    VectorStoreManager = None

try:
    from .retrieval import RetrievalChat

    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    RetrievalChat = None

try:
    from .document_processor import DocumentProcessor

    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSOR_AVAILABLE = False
    DocumentProcessor = None

__all__ = [
    "VectorStoreManager",
    "RetrievalChat",
    "DocumentProcessor",
    "VECTOR_STORE_AVAILABLE",
    "RETRIEVAL_AVAILABLE",
    "DOCUMENT_PROCESSOR_AVAILABLE",
]
