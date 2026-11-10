from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
from langchain_core.vectorstores import InMemoryVectorStore  # store chunks

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:4b"

text = Path("shop_handbook.md").read_text(encoding="utf-8") # string object

doc = Document(
    page_content=text,
    metadata={"source": "shop_handbook.md","author": "Cashfeet"}
) # document object


# 2.Chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents([doc]) # list of document objects


# 3.Embeddings

embeddings = init_embeddings(
                            model="nomic-embed-text:latest",
                            provider="ollama"
                             )


# 4. Vector Store

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)


# 5. RAG


llm = init_chat_model(
    model="gemma3:4b",
    model_provider="ollama",
    temperature=0.1
)


query = input("Query:  ").strip()

# Retrive
retriever = vector_store.similarity_search(query,k=3)

texts = []

for doc in retriever:
    texts.append(doc.page_content)

context = "\n\n".join(texts)


# Generate
messages = [
    {"role": "system", "content": "You are a helpful Clause shop Assistant!"},
    {"role": "user", "content": f"Answer the question based on the context below. If the answer is not contained within the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}"}
]


print(llm.invoke(messages, stream=True))
