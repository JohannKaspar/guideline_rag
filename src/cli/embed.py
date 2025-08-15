"""Embedding command implementation."""

from docling_core.types.doc.document import DoclingDocument
from tqdm import tqdm

from ..core import VectorStoreManager
from ..core.document_processor import DocumentProcessor
from ..utils import Config


def run():
    """Run the embedding process."""
    config = Config()

    # Get user-specific paths
    converted_dir, annotated_dir, _ = config.get_user_specific_paths()

    # Initialize components
    vector_store = VectorStoreManager()
    processor = DocumentProcessor()

    # Process all JSON files
    json_files = sorted(annotated_dir.glob("*.json"))[:3]  # Limit for testing

    for json_path in tqdm(json_files):
        filename = json_path.name
        pdf_name = filename.replace(".json", ".pdf")

        # Check if document already exists
        if vector_store.document_exists(pdf_name):
            print(f"Skipping {pdf_name}: already ingested.")
            continue

        print(f"Processing {pdf_name}…")

        # Load and chunk document
        doc = DoclingDocument.load_from_json(str(json_path))
        document_chunks = processor.chunk_document(doc, filename)

        # Add to vector store
        if document_chunks:
            vector_store.add_documents(document_chunks)
            print(f"Added {len(document_chunks)} chunks for {pdf_name}")


def main():
    """Main embedding function for backward compatibility."""
    run()


if __name__ == "__main__":
    main()
