"""Tests for main entry point."""

import sys
from unittest.mock import Mock, patch

import pytest

from src.main import main


class TestMain:
    """Test cases for main function."""

    def test_main_embed_command(self):
        """Test main function with embed command."""
        test_args = ["guideline-rag", "embed"]

        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.run_embed") as mock_run_embed,
        ):
            main()

            mock_run_embed.assert_called_once()

    def test_main_chat_command(self):
        """Test main function with chat command."""
        test_args = ["guideline-rag", "chat"]

        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.run_chat") as mock_run_chat,
        ):
            main()

            mock_run_chat.assert_called_once()

    def test_main_process_command(self):
        """Test main function with process command."""
        test_args = ["guideline-rag", "process"]

        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.run_process") as mock_run_process,
        ):
            main()

            mock_run_process.assert_called_once()

    def test_main_help_command(self, capsys):
        """Test main function with help command."""
        test_args = ["guideline-rag", "--help"]

        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "Guideline RAG System" in captured.out
        assert "embed" in captured.out
        assert "chat" in captured.out
        assert "process" in captured.out

    def test_main_invalid_command(self, capsys):
        """Test main function with invalid command."""
        test_args = ["guideline-rag", "invalid"]

        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "invalid choice" in captured.err.lower()

    def test_main_no_command(self, capsys):
        """Test main function with no command."""
        test_args = ["guideline-rag"]

        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "required" in captured.err.lower()

    def test_main_keyboard_interrupt(self, capsys):
        """Test main function handles keyboard interrupt."""
        test_args = ["guideline-rag", "embed"]

        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.run_embed", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Operation cancelled by user" in captured.out

    def test_main_general_exception(self, capsys):
        """Test main function handles general exceptions."""
        test_args = ["guideline-rag", "embed"]

        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.run_embed", side_effect=Exception("Test error")),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error running embed: Test error" in captured.out

    def test_main_cli_not_available(self, capsys):
        """Test main function when CLI is not available."""
        test_args = ["guideline-rag", "embed"]

        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.CLI_AVAILABLE", False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "CLI commands are not available" in captured.out

    def test_main_command_not_available(self, capsys):
        """Test main function when specific command is not available."""
        test_args = ["guideline-rag", "embed"]

        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.run_embed", None),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Embed command not available" in captured.out

    def test_main_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # Test that CLI_AVAILABLE is properly set based on imports
        from src.main import CLI_AVAILABLE

        # In test environment with mocked imports, this should be True
        # In real environment without dependencies, it would be False
        assert isinstance(CLI_AVAILABLE, bool)

    def test_main_with_missing_dependencies(self, capsys):
        """Test main behavior with missing dependencies."""
        test_args = ["guideline-rag", "embed"]

        # Simulate the case where dependencies are missing
        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.CLI_AVAILABLE", False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "CLI commands are not available due to missing dependencies" in captured.out
