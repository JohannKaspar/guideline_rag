"""Configuration management for the guideline RAG system."""

import os
from pathlib import Path


class Config:
    """Centralized configuration for the guideline RAG system."""

    # Directories
    BASE_DIR = Path(__file__).parent.parent.parent
    CONVERTED_DIR = BASE_DIR / "data" / "converted"
    ANNOTATED_DIR = BASE_DIR / "data" / "annotated"
    PDF_DIR = BASE_DIR / "pdfs"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    LOGS_DIR = BASE_DIR / "logs"

    # Database settings
    COLLECTION_NAME = "guidelines_chunks"

    # Model settings
    EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v3"

    # Processing settings
    MAX_CHUNK_TOKENS = 8192
    OVERLAP_TOKENS = 128
    MIN_PICTURE_WIDTH = 75
    MIN_PICTURE_HEIGHT = 75

    # API settings
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 1
    TEMPERATURE = 0

    # Batch processing
    BATCH_SIZE = 10
    PARALLEL_PROCESSES = 3

    @classmethod
    def get_user_specific_paths(cls) -> tuple[Path, Path, Path]:
        """Get user-specific paths based on environment configuration.
        
        Uses GUIDELINE_RAG_BASE_PATH environment variable if set,
        otherwise uses default paths from class constants.
        
        Example:
            export GUIDELINE_RAG_BASE_PATH="work/guideline_rag"
        """
        custom_base = os.getenv('GUIDELINE_RAG_BASE_PATH')
        if custom_base:
            base_path = Path(custom_base)
            return (
                base_path / "converted/",
                base_path / "annotated/",
                base_path / "pdfs/",
            )
        else:
            return (cls.CONVERTED_DIR, cls.ANNOTATED_DIR, cls.PDF_DIR)

    @classmethod
    def is_convert_only_mode(cls) -> bool:
        """Check if running in convert-only mode via environment variable.
        
        Uses CONVERT_ONLY_MODE environment variable.
        
        Example:
            export CONVERT_ONLY_MODE="true"
        """
        return os.getenv('CONVERT_ONLY_MODE', 'false').lower() == 'true'
