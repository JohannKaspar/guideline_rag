"""Document processing command implementation."""

import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from docling_core.types.doc.document import DoclingDocument
from tqdm import tqdm

from ..core.document_processor import DocumentProcessor
from ..utils import Config


def get_todo_files() -> list[str]:
    """Get list of files that need processing."""
    config = Config()
    converted_dir, annotated_dir, _ = config.get_user_specific_paths()

    # Get already processed files
    annotated = []
    converted = []

    if annotated_dir.exists():
        annotated = [f.stem for f in annotated_dir.glob("*.json")]

    if converted_dir.exists():
        converted = [f.stem for f in converted_dir.glob("*.json")]

    processed_pdfs = converted + annotated
    todo_files = []

    # Load AWMF guidelines if available
    awmf_file = Path("awmf.json")
    if awmf_file.exists():
        try:
            awmf_guidelines = json.load(open(awmf_file))
            for guideline in awmf_guidelines["records"]:
                for link in guideline["links"]:
                    if link["type"] == "longVersion":
                        file_name = os.path.basename(link["media"])
                        if Config.is_convert_only_mode():
                            if Path(file_name).stem not in processed_pdfs:
                                todo_files.append(file_name)
                        else:
                            if Path(file_name).stem not in annotated:
                                todo_files.append(file_name)
        except Exception as e:
            print(f"Warning: Could not load AWMF guidelines: {e}")

    return todo_files


def process_single_file(file_name: str) -> tuple[str, str | None]:
    """Process a single file. Returns (file_name, error_message)."""
    try:
        logging.info(f"Processing: {file_name}")

        config = Config()
        converted_dir, annotated_dir, pdf_dir = config.get_user_specific_paths()

        processor = DocumentProcessor()

        # Check if already converted
        json_file_name = file_name.replace(".pdf", ".json")
        converted_path = converted_dir / json_file_name

        if converted_path.exists():
            if Config.is_convert_only_mode():
                return file_name, None
            # Load existing converted document
            doc = DoclingDocument.load_from_json(str(converted_path))
        else:
            # Convert PDF
            pdf_path = pdf_dir / file_name
            if not pdf_path.exists():
                return file_name, f"PDF file not found: {pdf_path}"

            doc = processor.convert_document(pdf_path)

            # Save converted document
            converted_dir.mkdir(parents=True, exist_ok=True)
            doc.save_as_json(str(converted_path))

        # Annotate if not in convert-only mode
        if not Config.is_convert_only_mode():
            processor.annotate_document(doc)

            # Save annotated document
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = annotated_dir / json_file_name
            doc.save_as_json(str(annotated_path))

            # Remove converted file to save space
            if converted_path.exists():
                converted_path.unlink()

        return file_name, None

    except Exception as e:
        logging.error(f"Error processing {file_name}: {e}")
        return file_name, f"{type(e).__name__}: {e}"


def run():
    """Run document processing."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("convert.log"),
        ],
    )

    # Get files to process
    todo_files = get_todo_files()

    if not todo_files:
        print("No files to process.")
        return

    print(f"Found {len(todo_files)} files to process.")

    # Load or create status tracking
    status_file = Path("doc_status.csv")
    if status_file.exists():
        doc_status = pd.read_csv(status_file)
    else:
        doc_status = pd.DataFrame({"pdf_name": todo_files, "status": "pending"})

    # Process files with multiprocessing
    try:
        with Pool(processes=3) as pool:
            for pdf_name, error in tqdm(
                pool.imap_unordered(process_single_file, todo_files),
                total=len(todo_files),
                desc="Processing documents",
            ):
                # Update status
                mask = doc_status.pdf_name == pdf_name
                if error is None:
                    doc_status.loc[mask, "status"] = "ok"
                    print(f"✅ {pdf_name}")
                else:
                    doc_status.loc[mask, "status"] = error
                    print(f"❌ {pdf_name}: {error}")

                # Save status periodically
                doc_status.to_csv(status_file, index=False)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        doc_status.to_csv(status_file, index=False)

    except Exception as e:
        print(f"Error during processing: {e}")
        doc_status.to_csv(status_file, index=False)

    finally:
        # Final status save
        doc_status.to_csv(status_file, index=False)

        # Print summary
        success_count = (doc_status.status == "ok").sum()
        total_count = len(doc_status)
        print(f"\nProcessing complete: {success_count}/{total_count} files successful")


def main():
    """Main process function for backward compatibility."""
    run()


if __name__ == "__main__":
    main()
