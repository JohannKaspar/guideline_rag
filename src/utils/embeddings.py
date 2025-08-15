"""Centralized embedding model handling."""

import torch
from langchain.embeddings.base import Embeddings
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .config import Config


class HuggingFaceEmbeddings(Embeddings):
    """Wraps a HuggingFace model for LangChain embeddings."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or Config.EMBEDDING_MODEL_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading embedding model {self.model_id} on device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, task="retrieval.passage", device=self.device)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.model.encode([text], task="retrieval.query", device=self.device)[0]


def get_embeddings_model(model_id: str | None = None, device: str | None = None) -> HuggingFaceEmbeddings:
    """Factory function to get embeddings model."""
    return HuggingFaceEmbeddings(model_id=model_id, device=device)
