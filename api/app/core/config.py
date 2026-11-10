
from functools import lru_cache

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    # Document
    data_path: str = "shop_handbook.pdf"

    # Chat Model
    chat_provider: str = "ollama"
    chat_model: str = "gemma3:4b"
    temperature: float = 0.5

    # Embedding Model
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    # Retrieval
    top_k: int = 5

    # Chunking
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Qdrant
    collection_name: str = "shop_handbook"
    qdrant_path: str = "./qdrant_data"
    qdrant_url: str = "http://localhost:6333"

    # System Prompt
    system_prompt: str = (
        "You are a helpful assistant that answers questions "
        "about the shop handbook. Answer only using the provided "
        "handbook context. If the answer is not found in the "
        "context, respond with 'I don't know.' "
        "Do not invent information. Keep answers clear and accurate."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()



def get_embeddings() -> Embeddings:

    settings = get_settings()

    return init_embeddings(
        provider=settings.embed_provider,
        model=settings.embed_model,
    )



def get_qdrant_client(settings: Settings) -> QdrantClient:

    return QdrantClient(
        path=settings.qdrant_path
    )
