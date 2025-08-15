"""Tests for CLI process command."""

import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from src.cli.process import get_todo_files, process_single_file, run


class TestGetTodoFiles:
    """Test cases for get_todo_files function."""

    def test_get_todo_files_no_awmf_file(self, mock_config_paths):
        """Test get_todo_files when awmf.json doesn't exist."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        with patch("src.cli.process.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            with patch("pathlib.Path.exists", return_value=False):
                result = get_todo_files()

        assert result == []

    def test_get_todo_files_with_awmf_data(self, mock_config_paths, sample_awmf_data):
        """Test get_todo_files with AWMF data."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        with patch("src.cli.process.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("builtins.open", mock_open(read_data=json.dumps(sample_awmf_data))),
            ):
                result = get_todo_files()

        expected = ["guideline1.pdf", "guideline2.pdf"]
        assert result == expected

    def test_get_todo_files_with_processed_files(self, mock_config_paths, sample_awmf_data):
        """Test get_todo_files excludes already processed files."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        # Create some processed files
        (converted_dir / "guideline1.json").touch()
        (annotated_dir / "guideline2.json").touch()

        with patch("src.cli.process.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("builtins.open", mock_open(read_data=json.dumps(sample_awmf_data))),
            ):
                result = get_todo_files()

        # Should be empty since both files are already processed
        assert result == []

    def test_get_todo_files_convert_only_mode(self, mock_config_paths, sample_awmf_data):
        """Test get_todo_files in convert-only mode."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        # Create an annotated file
        (annotated_dir / "guideline1.json").touch()

        with patch("src.cli.process.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("builtins.open", mock_open(read_data=json.dumps(sample_awmf_data))),
                patch("src.cli.process.Config.is_convert_only_mode", return_value=True),
            ):
                result = get_todo_files()

        # In convert-only mode, should include guideline2.pdf but not guideline1.pdf
        expected = ["guideline2.pdf"]
        assert result == expected

    def test_get_todo_files_invalid_json(self, mock_config_paths):
        """Test get_todo_files with invalid JSON."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        with patch("src.cli.process.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("builtins.open", mock_open(read_data="invalid json")),
            ):
                result = get_todo_files()

        # Should return empty list on JSON error
        assert result == []


class TestProcessSingleFile:
    """Test cases for process_single_file function."""

    def test_process_single_file_success(self, mock_config_paths, mock_docling_document):
        """Test successful file processing."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        # Create a test PDF file
        test_pdf = pdf_dir / "test.pdf"
        test_pdf.touch()

        with (
            patch("src.cli.process.Config") as mock_config_class,
            patch("src.cli.process.DocumentProcessor") as mock_processor_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.convert_document.return_value = mock_docling_document
            mock_processor.annotate_document.return_value = None

            with patch("src.cli.process.Config.is_convert_only_mode", return_value=False):
                result = process_single_file("test.pdf")

        filename, error = result
        assert filename == "test.pdf"
        assert error is None

        # Verify processor methods were called
        mock_processor.convert_document.assert_called_once()
        mock_processor.annotate_document.assert_called_once()

    def test_process_single_file_convert_only(self, mock_config_paths, mock_docling_document):
        """Test file processing in convert-only mode."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        test_pdf = pdf_dir / "test.pdf"
        test_pdf.touch()

        with (
            patch("src.cli.process.Config") as mock_config_class,
            patch("src.cli.process.DocumentProcessor") as mock_processor_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.convert_document.return_value = mock_docling_document

            with patch("src.cli.process.Config.is_convert_only_mode", return_value=True):
                result = process_single_file("test.pdf")

        filename, error = result
        assert filename == "test.pdf"
        assert error is None

        # Should convert but not annotate in convert-only mode
        mock_processor.convert_document.assert_called_once()
        mock_processor.annotate_document.assert_not_called()

    def test_process_single_file_already_converted(self, mock_config_paths, mock_docling_document):
        """Test processing file that's already converted."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        # Create converted file
        converted_file = converted_dir / "test.json"
        converted_file.touch()

        with (
            patch("src.cli.process.Config") as mock_config_class,
            patch("src.cli.process.DocumentProcessor") as mock_processor_class,
            patch("src.cli.process.DoclingDocument.load_from_json") as mock_load,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            mock_load.return_value = mock_docling_document
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            with patch("src.cli.process.Config.is_convert_only_mode", return_value=False):
                result = process_single_file("test.pdf")

        filename, error = result
        assert filename == "test.pdf"
        assert error is None

        # Should load existing document, not convert
        mock_load.assert_called_once()
        mock_processor.convert_document.assert_not_called()
        mock_processor.annotate_document.assert_called_once()

    def test_process_single_file_missing_pdf(self, mock_config_paths):
        """Test processing non-existent PDF file."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        with (
            patch("src.cli.process.Config") as mock_config_class,
            patch("src.cli.process.DocumentProcessor") as mock_processor_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            result = process_single_file("nonexistent.pdf")

        filename, error = result
        assert filename == "nonexistent.pdf"
        assert error is not None and "PDF file not found" in error

    def test_process_single_file_exception(self, mock_config_paths):
        """Test processing file with exception."""
        converted_dir, annotated_dir, pdf_dir = mock_config_paths

        test_pdf = pdf_dir / "test.pdf"
        test_pdf.touch()

        with (
            patch("src.cli.process.Config") as mock_config_class,
            patch("src.cli.process.DocumentProcessor") as mock_processor_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_config.get_user_specific_paths.return_value = (
                converted_dir,
                annotated_dir,
                pdf_dir,
            )

            mock_processor_class.side_effect = Exception("Test error")

            result = process_single_file("test.pdf")

        filename, error = result
        assert filename == "test.pdf"
        assert error is not None and "Exception: Test error" in error


class TestRun:
    """Test cases for run function."""

    def test_run_no_files(self, capsys):
        """Test run with no files to process."""
        with patch("src.cli.process.get_todo_files", return_value=[]):
            run()

        captured = capsys.readouterr()
        assert "No files to process" in captured.out

    def test_run_keyboard_interrupt(self, capsys):
        """Test run handles keyboard interrupt gracefully."""
        with (
            patch("src.cli.process.get_todo_files", return_value=["test.pdf"]),
            patch("src.cli.process.Pool") as mock_pool,
            patch("src.cli.process.pd.DataFrame") as mock_df,
        ):
            mock_pool_instance = Mock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            mock_pool_instance.imap_unordered.side_effect = KeyboardInterrupt()

            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance

            run()

        captured = capsys.readouterr()
        assert "Processing interrupted by user" in captured.out

    def test_run_exception_handling(self, capsys):
        """Test run handles general exceptions."""
        with (
            patch("src.cli.process.get_todo_files", return_value=["test.pdf"]),
            patch("src.cli.process.Pool") as mock_pool,
        ):
            mock_pool.side_effect = Exception("Test error")

            run()

        captured = capsys.readouterr()
        assert "Error during processing" in captured.out
