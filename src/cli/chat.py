"""Chat command implementation."""

from ..core import RetrievalChat


def run():
    """Run the interactive chat."""
    chat = RetrievalChat()
    chat.run_interactive_chat()


def main():
    """Main chat function for backward compatibility."""
    run()


if __name__ == "__main__":
    main()
