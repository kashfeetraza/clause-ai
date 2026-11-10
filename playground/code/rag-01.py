# from pathlib import Path

# from langchain.chat_models import init_chat_model
# from langchain.embeddings import init_embeddings


# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
# from langchain_core.vectorstores import InMemoryVectorStore  # store chunks


# text = Path("shop_handbook.md").read_text(encoding="utf-8") # string object

# doc = Document(
#     page_content=text,
#     metadata={"source": "shop_handbook.md","author": "Cashfeet"}
# ) # document object


# # 2.Chunking

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=600,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", " ", ""]
# )

# chunks = text_splitter.split_documents([doc]) # list of document objects


# # 3.Embeddings

# embeddings = init_embeddings(
#                             model="nomic-embed-text:latest",
#                             provider="ollama"
#                              )


# # 4. Vector Store

# vector_store = InMemoryVectorStore.from_documents(
#     documents=chunks,
#     embedding=embeddings
# )


# # 5. RAG


# llm = init_chat_model(
#     model="gemma3:4b",
#     model_provider="ollama",
#     temperature=0.1
# )


# query = input("Query:  ").strip()

# # Retrive
# retriever = vector_store.similarity_search(query,k=3)

# texts = []

# for doc in retriever:
#     texts.append(doc.page_content)

# context = "\n\n".join(texts)


# # Generate
# messages = [
#     {"role": "system", "content": "You are a helpful Clause shop Assistant!"},
#     {"role": "user", "content": f"Answer the question based on the context below. If the answer is not contained within the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}"}
# ]


# print(llm.invoke(messages, stream=True))






from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


# ==========================================
# 1. DOCUMENT LOADING
# ==========================================

file_path = Path("shop_handbook.md")

if not file_path.exists():
    raise FileNotFoundError(
        f"File not found: {file_path.resolve()}"
    )

text = file_path.read_text(encoding="utf-8")

if not text.strip():
    raise ValueError("The shop handbook is empty!")

doc = Document(
    page_content=text,
    metadata={
        "source": "shop_handbook.md",
        "author": "Cashfeet"
    }
)

print("Document loaded successfully!")


# ==========================================
# 2. CHUNKING
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents([doc])

print("Total chunks:", len(chunks))


# ==========================================
# 3. EMBEDDINGS
# ==========================================

embeddings = init_embeddings(
    model="nomic-embed-text:latest",
    provider="ollama"
)

print("Embeddings model initialized!")


# ==========================================
# 4. VECTOR STORE
# ==========================================

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Vector store created successfully!")


# ==========================================
# 5. RAG - USER QUERY
# ==========================================

query = input("\nQuery: ").strip()

if not query:
    raise ValueError("Please enter a question!")


# ==========================================
# 6. RETRIEVAL
# ==========================================

retriever = vector_store.similarity_search(
    query,
    k=3
)

print("\n========== RETRIEVED DOCUMENTS ==========")

for i, item in enumerate(retriever, start=1):
    print(f"\nChunk {i}:")
    print(item.page_content)


# ==========================================
# 7. CONTEXT CREATION
# ==========================================

texts = []

for item in retriever:
    texts.append(item.page_content)

context = "\n\n".join(texts)

print("\n========== FINAL CONTEXT ==========")
print(context)


# ==========================================
# 8. CHAT MODEL
# ==========================================

llm = init_chat_model(
    model="gemma3:4b",
    model_provider="ollama",
    temperature=0.1
)


# ==========================================
# 9. PROMPT
# ==========================================

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful Clause Shop Assistant. "
            "Answer questions using only the provided context. "
            "Do not make up information. "
            "If the answer is not present in the context, "
            "say 'I don't know'."
        )
    },
    {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer based only on the context."
        )
    }
]


# ==========================================
# 10. GENERATION
# ==========================================

response = llm.invoke(messages)

print("\n========== FINAL ANSWER ==========")
print(response.content)

