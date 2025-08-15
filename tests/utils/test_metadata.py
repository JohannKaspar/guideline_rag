"""Tests for metadata processing utilities."""

from enum import Enum

import pytest

from src.utils.metadata import (
    clean_metadata_for_storage,
    flatten_metadata,
    normalize_whitespace,
)


class SampleEnum(Enum):
    """Sample enum for testing enum handling."""

    VALUE_A = "test_value_a"
    VALUE_B = "test_value_b"


class TestFlattenMetadata:
    """Test cases for flatten_metadata function."""

    def test_flatten_simple_dict(self):
        """Test flattening a simple dictionary."""
        metadata = {"key1": "value1", "key2": "value2"}
        result = flatten_metadata(metadata)

        expected = {"key1": "value1", "key2": "value2"}
        assert result == expected

    def test_flatten_nested_dict(self):
        """Test flattening nested dictionaries."""
        metadata = {"level1": {"level2": {"key": "value"}}}
        result = flatten_metadata(metadata)

        expected = {"level1.level2.key": "value"}
        assert result == expected

    def test_flatten_list(self):
        """Test flattening lists."""
        metadata = {"items": ["item1", "item2", "item3"]}
        result = flatten_metadata(metadata)

        expected = {"items[0]": "item1", "items[1]": "item2", "items[2]": "item3"}
        assert result == expected

    def test_flatten_tuple(self):
        """Test flattening tuples."""
        metadata = {"coords": (10, 20, 30)}
        result = flatten_metadata(metadata)

        expected = {"coords[0]": 10, "coords[1]": 20, "coords[2]": 30}
        assert result == expected

    def test_flatten_enum(self):
        """Test flattening enum values."""
        metadata = {"status": SampleEnum.VALUE_A}
        result = flatten_metadata(metadata)

        expected = {"status": "test_value_a"}
        assert result == expected

    def test_flatten_none_values(self):
        """Test that None values are excluded."""
        metadata = {"key1": "value1", "key2": None, "key3": "value3"}
        result = flatten_metadata(metadata)

        expected = {"key1": "value1", "key3": "value3"}
        assert result == expected

    def test_flatten_hash_values(self):
        """Test that hash values are converted to strings."""
        metadata = {"doc_hash": 12345, "file_hash": 67890}
        result = flatten_metadata(metadata)

        expected = {"doc_hash": "12345", "file_hash": "67890"}
        assert result == expected

    def test_flatten_complex_structure(self):
        """Test flattening a complex nested structure."""
        metadata = {
            "document": {
                "title": "Test Document",
                "authors": ["Author 1", "Author 2"],
                "metadata": {"created": "2024-01-01", "tags": ["tag1", "tag2"]},
            },
            "processing": {"status": SampleEnum.VALUE_B, "hash": 98765},
            "empty_field": None,
        }
        result = flatten_metadata(metadata)

        expected = {
            "document.title": "Test Document",
            "document.authors[0]": "Author 1",
            "document.authors[1]": "Author 2",
            "document.metadata.created": "2024-01-01",
            "document.metadata.tags[0]": "tag1",
            "document.metadata.tags[1]": "tag2",
            "processing.status": "test_value_b",
            "processing.hash": "98765",
        }
        assert result == expected

    def test_flatten_empty_dict(self):
        """Test flattening an empty dictionary."""
        metadata = {}
        result = flatten_metadata(metadata)

        assert result == {}

    def test_flatten_empty_list(self):
        """Test flattening empty lists."""
        metadata = {"empty_list": []}
        result = flatten_metadata(metadata)

        assert result == {}

    def test_flatten_mixed_types(self):
        """Test flattening with mixed data types."""
        metadata = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, "two", 3.0],
            "nested": {"inner_bool": False, "inner_list": ["a", "b"]},
        }
        result = flatten_metadata(metadata)

        expected = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list[0]": 1,
            "list[1]": "two",
            "list[2]": 3.0,
            "nested.inner_bool": False,
            "nested.inner_list[0]": "a",
            "nested.inner_list[1]": "b",
        }
        assert result == expected


class TestNormalizeWhitespace:
    """Test cases for normalize_whitespace function."""

    def test_normalize_multiple_spaces(self):
        """Test normalizing multiple spaces."""
        text = "This  has   multiple    spaces"
        result = normalize_whitespace(text)

        expected = "This has multiple spaces"
        assert result == expected

    def test_normalize_tabs(self):
        """Test normalizing tabs."""
        text = "This\thas\t\ttabs"
        result = normalize_whitespace(text)

        expected = "This has tabs"
        assert result == expected

    def test_normalize_mixed_whitespace(self):
        """Test normalizing mixed spaces and tabs."""
        text = "Mixed \t  spaces\t\tand   tabs"
        result = normalize_whitespace(text)

        expected = "Mixed spaces and tabs"
        assert result == expected

    def test_normalize_leading_trailing_preserved(self):
        """Test that leading and trailing whitespace is preserved."""
        text = "  leading and trailing  "
        result = normalize_whitespace(text)

        # Only internal multiple whitespace should be normalized
        expected = " leading and trailing "
        assert result == expected

    def test_normalize_newlines_preserved(self):
        """Test that newlines are preserved."""
        text = "Line 1\nLine  2\nLine   3"
        result = normalize_whitespace(text)

        expected = "Line 1\nLine 2\nLine 3"
        assert result == expected

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        text = ""
        result = normalize_whitespace(text)

        assert result == ""

    def test_normalize_only_whitespace(self):
        """Test normalizing string with only whitespace."""
        text = "   \t  \t   "
        result = normalize_whitespace(text)

        expected = " "
        assert result == expected

    def test_normalize_single_spaces_unchanged(self):
        """Test that single spaces remain unchanged."""
        text = "This has single spaces"
        result = normalize_whitespace(text)

        assert result == text


class TestCleanMetadataForStorage:
    """Test cases for clean_metadata_for_storage function."""

    def test_remove_image_keys(self):
        """Test that keys containing 'image' are removed."""
        metadata = {
            "title": "Document Title",
            "image_data": "base64_image_data",
            "thumbnail_image": "thumbnail_data",
            "IMAGE_URL": "http://example.com/image.jpg",
            "content": "Document content",
        }
        result = clean_metadata_for_storage(metadata)

        expected = {"title": "Document Title", "content": "Document content"}
        assert result == expected

    def test_remove_bytes_values(self):
        """Test that byte values are removed."""
        metadata = {
            "title": "Document Title",
            "binary_data": b"binary content",
            "text_data": "text content",
            "more_bytes": b"\x00\x01\x02",
        }
        result = clean_metadata_for_storage(metadata)

        expected = {"title": "Document Title", "text_data": "text content"}
        assert result == expected

    def test_case_insensitive_image_removal(self):
        """Test that image key removal is case insensitive."""
        metadata = {
            "Image": "data1",
            "IMAGE": "data2",
            "image": "data3",
            "ImAgE": "data4",
            "profile_image": "data5",
            "normal_key": "normal_value",
        }
        result = clean_metadata_for_storage(metadata)

        expected = {"normal_key": "normal_value"}
        assert result == expected

    def test_preserve_valid_data(self):
        """Test that valid data is preserved."""
        metadata = {
            "title": "Document Title",
            "author": "Author Name",
            "created_date": "2024-01-01",
            "page_count": 10,
            "tags": ["tag1", "tag2"],
            "metadata": {"nested": "value"},
            "is_processed": True,
            "score": 0.95,
        }
        result = clean_metadata_for_storage(metadata)

        # All data should be preserved as none contains 'image' or bytes
        assert result == metadata

    def test_empty_metadata(self):
        """Test cleaning empty metadata."""
        metadata = {}
        result = clean_metadata_for_storage(metadata)

        assert result == {}

    def test_mixed_removal(self):
        """Test removing both image keys and bytes together."""
        metadata = {
            "title": "Document",
            "cover_image": "image_data",
            "binary_content": b"binary_data",
            "description": "Text description",
            "thumbnail": b"thumbnail_bytes",
            "author": "Author Name",
        }
        result = clean_metadata_for_storage(metadata)

        expected = {
            "title": "Document",
            "description": "Text description",
            "author": "Author Name",
        }
        assert result == expected

    def test_partial_image_key_matches(self):
        """Test that partial matches of 'image' in keys are removed."""
        metadata = {
            "background_image_url": "url",
            "image_processing": "data",
            "normal_key": "value",
            "margin": "value2",  # Does not contain 'image'
        }
        result = clean_metadata_for_storage(metadata)

        # Only keys that don't contain 'image' should remain
        expected = {"normal_key": "value", "margin": "value2"}
        assert result == expected
