"""Main entry point for the guideline RAG system."""

import argparse
import sys

from .cli import run_chat, run_embed, run_process


def main():
    """Main entry point with command-line interface."""

    parser = argparse.ArgumentParser(description="Guideline RAG System")
    parser.add_argument(
        "command",
        choices=["embed", "chat", "process"],
        help="Command to run: embed documents, start chat, or process documents",
    )

    args = parser.parse_args()

    try:
        if args.command == "embed":
            if run_embed:
                run_embed()
            else:
                print("Embed command not available.")
                sys.exit(1)
        elif args.command == "chat":
            if run_chat:
                run_chat()
            else:
                print("Chat command not available.")
                sys.exit(1)
        elif args.command == "process":
            if run_process:
                run_process()
            else:
                print("Process command not available.")
                sys.exit(1)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running {args.command}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
