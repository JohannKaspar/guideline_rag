# Testing Guide

This directory contains comprehensive unit tests for the guideline RAG system.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_main.py            # Tests for main entry point
├── utils/
│   ├── test_config.py      # Tests for configuration management
│   └── test_metadata.py    # Tests for metadata utilities
├── cli/
│   └── test_process.py     # Tests for CLI process command
└── core/                   # Tests for core functionality (to be added)
```

## Running Tests

### Install Test Dependencies

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Or install development dependencies (includes test + linting)
uv pip install -e ".[dev]"
```


### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Categories

### Unit Tests
- **Configuration** (`tests/utils/test_config.py`): Tests configuration management and user-specific paths
- **Metadata** (`tests/utils/test_metadata.py`): Tests metadata processing utilities
- **CLI Process** (`tests/cli/test_process.py`): Tests document processing command
- **Main Entry** (`tests/test_main.py`): Tests main CLI entry point

### Test Features
- **Mocking**: Heavy dependencies (Docling, ChromaDB, etc.) are mocked
- **Fixtures**: Reusable test data and mock objects
- **Temporary Files**: Tests use temporary directories for file operations
- **Error Handling**: Tests cover both success and failure scenarios
- **Edge Cases**: Tests include boundary conditions and invalid inputs

## Writing New Tests

### Test File Naming
- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test functions should be named `test_*`

### Using Fixtures
```python
def test_with_temp_directory(temp_dir):
    """Test using temporary directory fixture."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()

def test_with_mock_config(mock_config_paths):
    """Test using mocked configuration paths."""
    converted_dir, annotated_dir, pdf_dir = mock_config_paths
    assert converted_dir.exists()
```

### Mocking External Dependencies
```python
@patch('src.utils.clients.get_chat_model')
def test_with_mocked_model(mock_get_model):
    """Test with mocked AI model."""
    mock_model = Mock()
    mock_get_model.return_value = mock_model
    # Test code here
```

## Test Coverage Goals

- **Utilities**: 90%+ coverage (high-priority, pure functions)
- **Core Logic**: 80%+ coverage (business logic)
- **CLI Commands**: 70%+ coverage (integration points)
- **Error Handling**: Focus on critical error paths

## Continuous Integration

The test suite is designed to run in CI environments with:
- Mocked heavy dependencies
- No external API calls
- Temporary file cleanup
- Deterministic test results

## Troubleshooting

### Import Errors
If you see import errors for `pytest` or other test dependencies:
```bash
uv pip install -e ".[test]"
```

### Mock Issues
If mocks aren't working as expected, check:
- Mock paths match the actual import paths
- Fixtures are properly applied
- Heavy imports are mocked in `conftest.py`

### Coverage Issues
If coverage is lower than expected:
- Check for untested error paths
- Add tests for edge cases
- Verify all public functions are tested
