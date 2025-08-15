"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config_paths(temp_dir):
    """Mock configuration paths to use temporary directory."""
    converted_dir = temp_dir / "converted"
    annotated_dir = temp_dir / "annotated"
    pdf_dir = temp_dir / "pdfs"

    # Create directories
    converted_dir.mkdir()
    annotated_dir.mkdir()
    pdf_dir.mkdir()

    with patch("src.utils.config.Config.get_user_specific_paths") as mock_paths:
        mock_paths.return_value = (converted_dir, annotated_dir, pdf_dir)
        yield converted_dir, annotated_dir, pdf_dir


@pytest.fixture
def mock_genai_client():
    """Mock GenAI Hub client."""
    with patch("src.utils.clients.get_proxy_client") as mock_client:
        mock_client.return_value = Mock()
        yield mock_client


@pytest.fixture
def mock_chat_model():
    """Mock chat model for testing."""
    mock_model = Mock()
    mock_model.invoke.return_value = "Mocked response"
    mock_model.batch.return_value = ["Mocked response 1", "Mocked response 2"]
    mock_model.with_retry.return_value = mock_model

    with patch("src.utils.clients.get_chat_model") as mock_get_model:
        mock_get_model.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_embeddings_model():
    """Mock embeddings model for testing."""
    mock_model = Mock()
    mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]

    with patch("src.utils.embeddings.get_embeddings_model") as mock_get_model:
        mock_get_model.return_value = mock_model
        yield mock_model


@pytest.fixture
def sample_awmf_data():
    """Sample AWMF guidelines data for testing."""
    return {
        "records": [
            {
                "title": "Test Guideline 1",
                "links": [
                    {
                        "type": "longVersion",
                        "media": "https://example.com/guideline1.pdf",
                    }
                ],
            },
            {
                "title": "Test Guideline 2",
                "links": [
                    {
                        "type": "longVersion",
                        "media": "https://example.com/guideline2.pdf",
                    }
                ],
            },
        ]
    }


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="This is the first chunk of medical content.",
            metadata={"source": "test_doc.pdf", "chunk_id": 0},
        ),
        Document(
            page_content="This is the second chunk with more medical information.",
            metadata={"source": "test_doc.pdf", "chunk_id": 1},
        ),
    ]


@pytest.fixture
def mock_docling_document():
    """Mock DoclingDocument for testing."""
    mock_doc = Mock()
    mock_doc.pictures = []
    mock_doc.tables = []
    mock_doc.export_to_markdown.return_value = "# Test Document\n\nThis is test content."
    mock_doc.save_as_json = Mock()

    return mock_doc


@pytest.fixture(autouse=True)
def mock_heavy_imports():
    """Mock heavy imports that might not be available in test environment."""
    with patch.dict(
        "sys.modules",
        {
            "docling": Mock(),
            "docling_core": Mock(),
            "docling.datamodel.base_models": Mock(),
            "docling.datamodel.pipeline_options": Mock(),
            "docling.document_converter": Mock(),
            "docling_core.types.doc.document": Mock(),
            "chromadb": Mock(),
            "torch": Mock(),
            "transformers": Mock(),
        },
    ):
        yield


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    test_env = {
        "GENAI_HUB_URL": "https://test-genai-hub.com",
        "GENAI_HUB_CLIENT_ID": "test-client-id",
        "GENAI_HUB_CLIENT_SECRET": "test-client-secret",
    }

    with patch.dict(os.environ, test_env):
        yield test_env
