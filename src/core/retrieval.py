"""Retrieval and chat functionality."""

from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

from ..utils import get_chat_model
from .vector_store import VectorStoreManager


class RetrievalChat:
    """Handles retrieval-based chat functionality."""

    def __init__(self, vector_store_manager: VectorStoreManager | None = None):
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.chat_model = get_chat_model("gpt-4.1")

        # Setup retriever
        self.retriever = self.vector_store_manager.get_retriever(search_type="similarity", k=5)

        # Setup conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model, retriever=self.retriever, return_source_documents=False
        )

    def chat(self, query: str, chat_history: list[tuple[str, str]] | None = None) -> str:
        """Process a chat query and return response."""
        if chat_history is None:
            chat_history = []

        result = self.qa_chain({"question": query, "chat_history": chat_history})
        return result["answer"]

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Perform similarity search."""
        return self.vector_store_manager.similarity_search(query, k=k)

    def run_interactive_chat(self):
        """Run interactive chat session."""
        chat_history: list[tuple[str, str]] = []
        print("Chatbot ready. Type 'exit' to quit.\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            answer = self.chat(query, chat_history)
            print(f"Bot: {answer}\n")
            chat_history.append((query, answer))
