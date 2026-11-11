from app.services.ingestion import ingest
from app.services.chat import RAGServices


def main():

    # Step 1: Ingest document
    total_chunks = ingest()

    print(f"\nIngestion completed: {total_chunks} chunks")

    # Step 2: Initialize RAG service
    rag = RAGServices()

    try:
        # Step 3: Ask questions
        while True:

            question = input(
                "\nAsk your question (type 'exit' to quit): "
            ).strip()

            if question.lower() == "exit":
                break

            try:
                result = rag.ask(question)

                print("\nAnswer:")
                print(result.answer)

                print("\nSources:")

                for source in result.sources:
                    print(source.model_dump())

            except ValueError as e:
                print(f"Error: {e}")

    finally:
        rag.close()
        print("\nRAG service closed.")


if __name__ == "__main__":
    main()
