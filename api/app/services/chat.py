# from httpx2 import query
# from langchain.chat_models import init_chat_model
# from langchin_qdrant import QdrantVectorStore

# from core.config import get_embeddings, get_qdrant_client, get_settings

# from pydantic import BaseModel

# NO_ANSWER = "I don't Know. The handbook doesnot cover that."

# class Source(BaseModel):
#     source: str
#     page: int | None = None
#     score: float

# class Answer(BaseModel):
#     answer: str
#     sources: list[Source]


# class RAGServices():
#     def __init__(self):
#         self.settings = get_settings
#         self.client = get_qdrant_client(settings=self.settings)
#         if not self.client:
#             raise RuntimeError(f"Colletion {self.settings.collection_name} Not Founded!")

#         self.store = QdrantVectorStore(
#             client=self.settings.qdrant_path,
#             collection_name=self.settings.collection_name,
#             embedding=get_embeddings(self.settings)
#         )
#         self.llm = init_chat_model(
#             model=self.settings.chat_model,
#             model_provider=self.settings.chat_provider,
#             temperature=self.settings.chat_temperature
#         )

#     def ask(self, question:str) -> None:
#         question = question.strp()
#         if not question:
#             raise ValueError("Question must be not empty")
#         results = self.store.similarity_search_with_score(
#             query=question,
#             k=self.settings.top_k
#         )

#         if not results:
#             return Answer(answer=NO_ANSWER, source=[])

#         context = "\n\n.join(doc.page_content for doc, score in results)"

#         message = [
#             {"role": "system", "content": self.settings.system_prompt}
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}}

#         ]

#     return Answer(
#         answer=self.llm.invoke(input=messages).text,
#         source=[
#             source=doc.metadata.get("sources", "unknown"),
#             page=doc.metadata.get("page"),
#             score=round(score, 4)
#             for doc, score in results
#         ]
#     )





#     def close(self) -> None:
#         pass

# def main():
#     rag = RAGServices()
#     try:
#         query = input("Enter your question: ").strip()
#         result = rag.ask(query)
#     finally:
#         rag.close()





from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore

from app.core.config import (
    get_embeddings,
    get_qdrant_client,
    get_settings,
)

from pydantic import BaseModel


NO_ANSWER = "I don't know. The handbook does not cover that."


class Source(BaseModel):
    source: str
    page: int | None = None
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class RAGServices():

    def __init__(self):

        # 1. Load settings
        self.settings = get_settings()

        # 2. Connect to Qdrant
        self.client = get_qdrant_client(settings=self.settings)

        if not self.client.collection_exists(
            collection_name=self.settings.collection_name
        ):
            self.client.close()
            raise RuntimeError(
                f"Collection {self.settings.collection_name} not found!"
            )

        # 3. Initialize vector store
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=get_embeddings()
        )

        # 4. Initialize LLM
        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.temperature
        )

    # 5. Ask question
    def ask(self, question: str) -> Answer:

        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty")

        # 6. Search relevant documents
        results = self.store.similarity_search_with_score(
            query=question,
            k=self.settings.top_k
        )

        if not results:
            return Answer(
                answer=NO_ANSWER,
                sources=[]
            )

        # 7. Prepare context
        context = "\n\n".join(
            doc.page_content for doc, score in results
        )

        # 8. Prepare messages
        messages = [
            {
                "role": "system",
                "content": self.settings.system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]

        # 9. Generate answer
        response = self.llm.invoke(messages)

        # 10. Return answer with sources
        return Answer(
            answer=response.content,
            sources=[
                Source(
                    source=doc.metadata.get("source", "unknown"),
                    page=(
                        doc.metadata.get("page") + 1
                        if doc.metadata.get("page") is not None
                        else None
                    ),
                    score=round(float(score), 4)
                )
                for doc, score in results
            ]
        )

    # 11. Close connection
    def close(self) -> None:
        self.client.close()


def main():

    rag = RAGServices()

    try:
        question = input("Enter your question: ").strip()

        result = rag.ask(question)

        print("\nAnswer:")
        print(result.answer)

        print("\nSources:")

        for source in result.sources:
            print(source.model_dump())

    finally:
        rag.close()


if __name__ == "__main__":
    main()
