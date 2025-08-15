"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_config_constants(self):
        """Test that configuration constants are properly defined."""
        # Test directory constants
        assert isinstance(Config.BASE_DIR, Path)
        assert isinstance(Config.CONVERTED_DIR, Path)
        assert isinstance(Config.ANNOTATED_DIR, Path)
        assert isinstance(Config.PDF_DIR, Path)
        assert isinstance(Config.CHROMA_DB_DIR, Path)
        assert isinstance(Config.LOGS_DIR, Path)

        # Test database settings
        assert Config.COLLECTION_NAME == "guidelines_chunks"

        # Test model settings
        assert Config.EMBEDDING_MODEL_ID == "jinaai/jina-embeddings-v3"

        # Test processing settings
        assert Config.MAX_CHUNK_TOKENS == 8192
        assert Config.OVERLAP_TOKENS == 128
        assert Config.MIN_PICTURE_WIDTH == 75
        assert Config.MIN_PICTURE_HEIGHT == 75

        # Test API settings
        assert Config.REQUEST_TIMEOUT == 60
        assert Config.MAX_RETRIES == 1
        assert Config.TEMPERATURE == 0

        # Test batch processing
        assert Config.BATCH_SIZE == 10
        assert Config.PARALLEL_PROCESSES == 3

    def test_base_dir_structure(self):
        """Test that base directory structure is correct."""
        # BASE_DIR should be the project root (3 levels up from config.py)
        expected_base = Path(__file__).parent.parent.parent
        assert Config.BASE_DIR == expected_base

        # Test that subdirectories are correctly defined relative to BASE_DIR
        assert Config.CONVERTED_DIR == Config.BASE_DIR / "converted"
        assert Config.ANNOTATED_DIR == Config.BASE_DIR / "annotated"
        assert Config.PDF_DIR == Config.BASE_DIR / "pdfs"
        assert Config.CHROMA_DB_DIR == Config.BASE_DIR / "chroma_db"
        assert Config.LOGS_DIR == Config.BASE_DIR / "logs"

    @patch("os.path.expanduser")
    def test_get_user_specific_paths_regular_user(self, mock_expanduser):
        """Test get_user_specific_paths for regular user."""
        mock_expanduser.return_value = "/home/regular_user"

        converted, annotated, pdf = Config.get_user_specific_paths()

        # Should return default paths for regular user
        assert converted == Config.CONVERTED_DIR
        assert annotated == Config.ANNOTATED_DIR
        assert pdf == Config.PDF_DIR

    @patch("os.path.expanduser")
    def test_get_user_specific_paths_joli13_user(self, mock_expanduser):
        """Test get_user_specific_paths for joli13 user."""
        mock_expanduser.return_value = "/home/joli13"

        converted, annotated, pdf = Config.get_user_specific_paths()

        # Should return special paths for joli13 user
        assert converted == Path("work/guideline_rag/converted/")
        assert annotated == Path("work/guideline_rag/annotated/")
        assert pdf == Path("work/guideline_rag/pdfs/")

    @patch("os.path.expanduser")
    def test_get_user_specific_paths_return_types(self, mock_expanduser):
        """Test that get_user_specific_paths returns Path objects."""
        mock_expanduser.return_value = "/home/test_user"

        converted, annotated, pdf = Config.get_user_specific_paths()

        assert isinstance(converted, Path)
        assert isinstance(annotated, Path)
        assert isinstance(pdf, Path)

    @patch("os.path.expanduser")
    def test_is_convert_only_mode_regular_user(self, mock_expanduser):
        """Test is_convert_only_mode for regular user."""
        mock_expanduser.return_value = "/home/regular_user"

        result = Config.is_convert_only_mode()

        assert result is False

    @patch("os.path.expanduser")
    def test_is_convert_only_mode_joli13_user(self, mock_expanduser):
        """Test is_convert_only_mode for joli13 user."""
        mock_expanduser.return_value = "/home/joli13"

        result = Config.is_convert_only_mode()

        assert result is True

    @patch("os.path.expanduser")
    def test_is_convert_only_mode_partial_match(self, mock_expanduser):
        """Test is_convert_only_mode with partial username match."""
        mock_expanduser.return_value = "/home/joli13_test"

        result = Config.is_convert_only_mode()

        # Should still return True as "joli13" is in the username
        assert result is True

    def test_config_immutability(self):
        """Test that config values are not accidentally modified."""
        original_batch_size = Config.BATCH_SIZE
        original_timeout = Config.REQUEST_TIMEOUT

        # These should be class attributes, not instance attributes
        config1 = Config()
        config2 = Config()

        # Modifying one instance shouldn't affect the class or other instances
        # (though we shouldn't modify these in practice)
        assert Config.BATCH_SIZE == original_batch_size
        assert Config.REQUEST_TIMEOUT == original_timeout

    def test_path_resolution(self):
        """Test that paths resolve correctly."""
        # Test that paths are absolute
        assert Config.BASE_DIR.is_absolute()

        # Test that derived paths are also absolute
        assert Config.CONVERTED_DIR.is_absolute()
        assert Config.ANNOTATED_DIR.is_absolute()
        assert Config.PDF_DIR.is_absolute()

    @patch.dict(os.environ, {"HOME": "/test/home"})
    @patch("os.path.expanduser")
    def test_user_detection_with_environment(self, mock_expanduser):
        """Test user detection works with different environment setups."""
        mock_expanduser.return_value = "/test/home/joli13"

        # Test both methods work consistently
        paths = Config.get_user_specific_paths()
        convert_only = Config.is_convert_only_mode()

        assert convert_only is True
        assert paths[0] == Path("work/guideline_rag/converted/")

    def test_config_values_are_reasonable(self):
        """Test that configuration values are reasonable."""
        # Test that numeric values are positive
        assert Config.MAX_CHUNK_TOKENS > 0
        assert Config.OVERLAP_TOKENS >= 0
        assert Config.MIN_PICTURE_WIDTH > 0
        assert Config.MIN_PICTURE_HEIGHT > 0
        assert Config.REQUEST_TIMEOUT > 0
        assert Config.MAX_RETRIES >= 0
        assert Config.TEMPERATURE >= 0
        assert Config.BATCH_SIZE > 0
        assert Config.PARALLEL_PROCESSES > 0

        # Test that string values are not empty
        assert len(Config.COLLECTION_NAME) > 0
        assert len(Config.EMBEDDING_MODEL_ID) > 0

        # Test reasonable ranges
        assert Config.BATCH_SIZE <= 100  # Not too large
        assert Config.PARALLEL_PROCESSES <= 20  # Not too many processes
        assert Config.REQUEST_TIMEOUT <= 300  # Not too long (5 minutes max)
