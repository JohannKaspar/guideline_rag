"""Simple focused tests for core functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.config import Config
from src.utils.metadata import flatten_metadata, normalize_whitespace, clean_metadata_for_storage


class TestConfig:
    """Simple tests for configuration."""

    def test_config_constants(self):
        """Test that config constants are defined."""
        assert hasattr(Config, "CONVERTED_DIR")
        assert hasattr(Config, "ANNOTATED_DIR")
        assert hasattr(Config, "PDF_DIR")

    def test_is_convert_only_mode(self):
        """Test convert-only mode detection."""
        with patch("os.path.expanduser", return_value="/home/joli13"):
            assert Config.is_convert_only_mode() is True

        with patch("os.path.expanduser", return_value="/home/other_user"):
            assert Config.is_convert_only_mode() is False


class TestMetadataUtils:
    """Simple tests for metadata utilities."""

    def test_flatten_simple(self):
        """Test basic flattening."""
        data = {"a": 1, "b": {"c": 2}}
        result = flatten_metadata(data)
        expected = {"a": 1, "b.c": 2}
        assert result == expected

    def test_normalize_whitespace_basic(self):
        """Test basic whitespace normalization."""
        text = "hello  world"
        result = normalize_whitespace(text)
        assert result == "hello world"

    def test_clean_metadata_basic(self):
        """Test basic metadata cleaning."""
        data = {"title": "test", "image_data": "remove_me"}
        result = clean_metadata_for_storage(data)
        expected = {"title": "test"}
        assert result == expected


class TestImports:
    """Test that imports work correctly."""

    def test_config_import(self):
        """Test config can be imported."""
        from src.utils.config import Config

        assert Config is not None

    def test_metadata_import(self):
        """Test metadata utils can be imported."""
        from src.utils.metadata import flatten_metadata

        assert flatten_metadata is not None

    def test_main_import(self):
        """Test main can be imported."""
        from src.main import main

        assert main is not None
