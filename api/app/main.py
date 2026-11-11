
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from app.services.chat import RAGServices, Answer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.rag = RAGServices()
    yield
    app.state.rag.close()


app = FastAPI(title="Clause AI APIs", lifespan=lifespan)


class AskQuestion(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=2000
        )
    ]


@app.get("/")
def health():
    return {"status": "OK"}


@app.post("/ask", response_model=Answer)
def ask_query(body: AskQuestion, req: Request):
    return req.app.state.rag.ask(body.question)
