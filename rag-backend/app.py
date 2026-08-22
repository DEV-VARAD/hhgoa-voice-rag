# Run locally with: uvicorn app:app --reload
"""FastAPI interface for the guarded multilingual RAG pipeline."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from embed_index import DEFAULT_MODEL_NAME, load_index
from harness import DEFAULT_SCORE_THRESHOLD, process_query


INDEX_PATH = os.environ.get("INDEX_PATH", "index.faiss")
METADATA_PATH = os.environ.get("METADATA_PATH", "metadata.json")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
# load_index reads the model saved alongside the index; this setting documents
# the expected embedding-model default for deployment configuration.
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", DEFAULT_MODEL_NAME)


class QueryRequest(BaseModel):
    """Payload accepted by the frontend query endpoint."""

    text: str = Field(..., description="Question to send through the RAG pipeline")
    language: str | None = Field(default=None, description="Optional caller language hint")
    query_id: str | None = Field(default=None, description="Optional caller request identifier")


class Source(BaseModel):
    """A retrieved source returned with a grounded answer."""

    text: str
    language: str | None = None
    score: float


class QueryResponse(BaseModel):
    """Stable frontend response shape for success and refusal outcomes."""

    answer: str | None
    grounded: bool
    refused: bool
    refusal_reason: str | None
    sources: list[Source]
    latency_ms: dict[str, float]
    model_used: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and sentence-transformer model once at startup."""
    faiss_index, metadata_list, embed_model = load_index(INDEX_PATH, METADATA_PATH)
    app.state.faiss_index = faiss_index
    app.state.metadata_list = metadata_list
    app.state.embed_model = embed_model
    yield


app = FastAPI(title="Hindi-English RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _response_from_harness(result: dict[str, Any]) -> QueryResponse:
    """Translate harness statuses into the API's stable success/refusal schema."""
    status = result.get("status")
    answer_text = result.get("answer") or ""
    is_llm_refusal = answer_text.strip() == "I don't have enough information to answer that"
    if status == "success" and not is_llm_refusal:
        return QueryResponse(
            answer=result.get("answer"),
            grounded=True,
            refused=False,
            refusal_reason=None,
            sources=result.get("sources", []),
            latency_ms=result.get("latency_ms", {}),
            model_used=result.get("model_used", ""),
        )
    if status == "success" and is_llm_refusal:
        return QueryResponse(
            answer=result.get("answer"),
            grounded=False,
            refused=True,
            refusal_reason="llm_insufficient_context",
            sources=result.get("sources", []),
            latency_ms=result.get("latency_ms", {}),
            model_used=result.get("model_used", ""),
        )
    if status in {"rejected", "no_relevant_context"}:
        return QueryResponse(
            answer=result.get("answer"),
            grounded=False,
            refused=True,
            refusal_reason=result.get("reason", status),
            sources=[],
            latency_ms=result.get("latency_ms", {}),
            model_used=result.get("model_used", ""),
        )
    raise HTTPException(
        status_code=500,
        detail=f"RAG pipeline error: {result.get('reason', 'unknown_error')}",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a lightweight readiness response for frontend availability checks."""
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Run one validated query against the startup-loaded RAG resources."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty or whitespace only")
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server.",
        )

    try:
        result = process_query(
            request.text,
            app.state.faiss_index,
            app.state.metadata_list,
            app.state.embed_model,
            GROQ_API_KEY,
            GROQ_MODEL,
            top_k=5,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail="RAG pipeline request failed.") from error
    return _response_from_harness(result)
