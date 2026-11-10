from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
from langchain_core.vectorstores import InMemoryVectorStore  # store chunks

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:4b"

# NEW: is se kam score wale chunks ko hum "irrelevant" samjhenge
# Pehle scores print karke dekho, phir isko adjust karo
SCORE_THRESHOLD = 0.5

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
retriever = vector_store.similarity_search_with_score(query,k=3)

# NEW: scores print karo taake tum threshold sahi choose kar sako
for doc, score in retriever:
    print(f"Score: {score:.3f} | {doc.page_content[:60]!r}")

texts = []

for doc, _ in retriever:
    # CHANGED: ab score use ho raha hai, sirf relevant chunks hi context mein jayenge
    if _ >= SCORE_THRESHOLD:
        texts.append(doc.page_content)

# NEW: agar koi bhi chunk relevant nahi, to LLM ko call kiye baghair hi jawab do
if not texts:
    print("I don't know")
    raise SystemExit

context = "\n\n".join(texts)


# Generate
messages = [
    {"role": "system", "content": "You are a helpful Clause shop Assistant!"},
    {"role": "user", "content": f"Answer the question based on the context below. If the answer is not contained within the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}"}
]


# CHANGED: pehle poora message object print ho raha tha, ab sirf jawab (.content)
print(llm.invoke(messages).content)


