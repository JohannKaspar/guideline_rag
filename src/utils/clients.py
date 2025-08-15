"""Client setup utilities for GenAI Hub and other services."""

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings

from .config import Config


def get_genai_hub_client():
    """Get GenAI Hub proxy client."""
    return get_proxy_client("gen-ai-hub")


def get_chat_model(model_name: str = "gpt-4.1-mini", temperature: float | None = None):
    """Get ChatOpenAI model with GenAI Hub proxy."""
    proxy_client = get_genai_hub_client()

    return ChatOpenAI(
        proxy_model_name=model_name,
        proxy_client=proxy_client,
        temperature=temperature or Config.TEMPERATURE,
        max_retries=Config.MAX_RETRIES,
        request_timeout=Config.REQUEST_TIMEOUT,
    )


def get_openai_embeddings(model_name: str = "text-embedding-3-small"):
    """Get OpenAI embeddings with GenAI Hub proxy."""
    proxy_client = get_genai_hub_client()

    return OpenAIEmbeddings(
        proxy_client=proxy_client,
        proxy_model_name=model_name,
        max_retries=Config.MAX_RETRIES,
        request_timeout=Config.REQUEST_TIMEOUT,
    )
