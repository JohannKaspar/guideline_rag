"""Vector store operations for the guideline RAG system."""

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..utils import Config, get_embeddings_model


class VectorStoreManager:
    """Manages vector store operations."""

    def __init__(self, db_dir: Path | None = None, collection_name: str | None = None):
        self.db_dir = str(db_dir or Config.CHROMA_DB_DIR)
        self.collection_name = collection_name or Config.COLLECTION_NAME

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        # Initialize embeddings
        self.embeddings = get_embeddings_model()

        # Initialize LangChain vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_dir,
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return

        self.vector_store.add_documents(documents)

    def document_exists(self, filename: str) -> bool:
        """Check if a document has already been ingested."""
        existing = self.collection.get(where={"origin.filename": filename})
        return bool(existing.get("ids"))

    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """Get a retriever for the vector store."""
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Perform similarity search."""
        return self.vector_store.similarity_search(query, k=k)
