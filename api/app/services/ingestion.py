
# from pathlib import Path

# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client.models import Distance, VectorParams

# from app.core.config import (
#     get_embeddings,
#     get_qdrant_client,
#     get_settings,
# )


# def load(path: Path) -> list[Document]:
#     """Load PDF documents and extract their text."""
#     loader = PyPDFLoader(str(path))
#     return loader.load()


# def ingest(path: str | None = None) -> int:
#     """Load, split, embed, and store PDF documents."""

#     settings = get_settings()

#     file = Path(path or settings.data_path)

#     if not file.exists():
#         raise FileNotFoundError(f"PDF file not found: {file}")

#     # 1. Load PDF
#     documents = load(file)

#     # 2. Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=settings.chunk_size,
#         chunk_overlap=settings.chunk_overlap,
#         separators=["\n\n", "\n", " ", ""],
#     )

#     chunks = text_splitter.split_documents(documents)

#     print(f"Loaded {len(documents)} pages")
#     print(f"Created {len(chunks)} chunks")

#     # 3. Create embeddings
#     embed = get_embeddings(settings)

#     # 4. Connect to Qdrant
#     client = get_qdrant_client(settings)

#     try:
#         # Delete existing collection before recreating it
#         if client.collection_exists(
#             collection_name=settings.collection_name
#         ):
#             client.delete_collection(
#                 collection_name=settings.collection_name
#             )

#         # Create a new collection
#         client.create_collection(
#             collection_name=settings.collection_name,
#             vectors_config=VectorParams(
#                 size=settings.embedding_dim,
#                 distance=Distance.COSINE,
#             ),
#         )

#         # 5. Create vector store
#         vector_store = QdrantVectorStore(
#             client=client,
#             collection_name=settings.collection_name,
#             embedding=embed,
#         )

#         # 6. Store document chunks
#         vector_store.add_documents(chunks)

#         print("Documents successfully stored in Qdrant!")

#         return len(chunks)

#     finally:
#         client.close()


# if __name__ == "__main__":
#     total_chunks = ingest()
#     print(f"Total chunks indexed: {total_chunks}")




from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from app.core.config import (
    get_embeddings,
    get_qdrant_client,
    get_settings,
)


def load(path: Path) -> list[Document]:
    """Load PDF documents and extract their text."""
    loader = PyPDFLoader(str(path))
    return loader.load()


def ingest(path: str | None = None) -> int:
    """Load, split, embed, and store PDF documents."""

    settings = get_settings()
    file = Path(path or settings.data_path)

    if not file.exists():
        raise FileNotFoundError(f"PDF file not found: {file}")

    # 1. Load PDF
    documents = load(file)

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Loaded {len(documents)} pages")
    print(f"Created {len(chunks)} chunks")

    # 3. Create embeddings
    embed = get_embeddings()

    # 4. Connect to Qdrant
    client = get_qdrant_client(settings)

    try:
        # Delete existing collection
        if client.collection_exists(
            collection_name=settings.collection_name
        ):
            client.delete_collection(
                collection_name=settings.collection_name
            )

        # Create new collection
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embed_dim,
                distance=Distance.COSINE,
            ),
        )

        # 5. Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embed,
        )

        # 6. Store document chunks
        vector_store.add_documents(chunks)

        print("Documents successfully stored in Qdrant!")

        return len(chunks)

    finally:
        client.close()


if __name__ == "__main__":
    total_chunks = ingest()
    print(f"Total chunks indexed: {total_chunks}")
