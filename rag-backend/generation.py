"""Grounded answer generation over chunks returned by :mod:`retrieval`.

Groq is used through its OpenAI-compatible Python SDK. The system instruction
requires answers to be grounded exclusively in retrieved context.
"""

from __future__ import annotations

import argparse
import os
import time

from embed_index import load_index
from retrieval import retrieve


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough information to answer that"


def _build_context_prompt(query: str, retrieved_chunks: list[dict]) -> tuple[str, list[str]]:
    """Format retrieval output into labeled, model-readable context blocks."""
    context_texts: list[str] = []
    blocks: list[str] = []
    for position, chunk in enumerate(retrieved_chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        metadata = chunk.get("metadata", {})
        language = metadata.get("language", "unknown")
        score = float(chunk.get("score", 0.0))
        context_texts.append(text)
        blocks.append(
            f"[CONTEXT {position} | language={language} | similarity_score={score:.4f}]\n{text}"
        )

    context = "\n\n".join(blocks) if blocks else "[NO CONTEXT RETRIEVED]"
    prompt = f"User query:\n{query}\n\nRetrieved context:\n{context}"
    return prompt, context_texts


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    groq_api_key: str,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> dict:
    """Generate a context-grounded answer with Groq and retry request failures.

    At most three API attempts are made: the initial request and two retries
    after 0.5 and 1.0 seconds. Retries are limited to connection, timeout,
    rate-limit, and server failures; a successful response with empty content
    is treated as an error and is not retried.

    Returns:
        A dictionary with ``answer``, ``model_used``, ``latency_ms``, and the
        list of source strings in ``raw_context_used``.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(groq_api_key, str) or not groq_api_key.strip():
        raise ValueError(
            "A Groq API key is required. Pass --groq_api_key or set GROQ_API_KEY."
        )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    try:
        from groq import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            Groq,
            InternalServerError,
            RateLimitError,
        )
    except ImportError as error:
        raise ImportError("generation.py requires the groq package. Install it with: pip install groq") from error

    user_prompt, raw_context_used = _build_context_prompt(query, retrieved_chunks)
    system_prompt = (
        "You are a retrieval-augmented assistant. Answer ONLY with facts explicitly "
        "supported by the retrieved context. Do not use outside knowledge, infer missing "
        "facts, or invent details. If the context does not contain a relevant answer, you "
        f"MUST respond with exactly: {INSUFFICIENT_CONTEXT_MESSAGE}"
    )
    client = Groq(api_key=groq_api_key)
    retryable_errors = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

    generation_started = time.perf_counter()
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            answer = completion.choices[0].message.content
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("Groq returned a successful response without answer text.")
            return {
                "answer": answer.strip(),
                "model_used": model_name,
                "latency_ms": (time.perf_counter() - generation_started) * 1000,
                "raw_context_used": raw_context_used,
            }
        except retryable_errors as error:
            if attempt == 2:
                raise RuntimeError("Groq request failed after 3 attempts. Please try again later.") from error
            time.sleep((attempt + 1) * 0.5)
        except AuthenticationError as error:
            raise ValueError("The Groq API key is missing, invalid, or unauthorized.") from error
        except APIStatusError as error:
            raise RuntimeError(f"Groq rejected the request: {error}") from error

    raise RuntimeError("Groq generation failed unexpectedly.")  # Defensive, loop always returns or raises.


def main() -> None:
    """Retrieve context, generate a grounded answer, and print latency totals."""
    parser = argparse.ArgumentParser(description="Generate a grounded Groq answer from a FAISS index.")
    parser.add_argument("--index_path", default="index.faiss")
    parser.add_argument("--metadata_path", default="metadata.json")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Embedding model label; load_index uses the model stored with the index.",
    )
    parser.add_argument("--groq_model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--groq_api_key", default=os.environ.get("GROQ_API_KEY"))
    args = parser.parse_args()

    try:
        faiss_index, metadata_list, model = load_index(args.index_path, args.metadata_path)
        retrieval_started = time.perf_counter()
        retrieved_chunks = retrieve(args.query, faiss_index, metadata_list, model, args.top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        generated = generate_answer(
            args.query, retrieved_chunks, args.groq_api_key, args.groq_model
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    generation_ms = generated["latency_ms"]
    print("\nAnswer:\n" + generated["answer"])
    print("\nLatency breakdown:")
    print(f"  retrieval_ms: {retrieval_ms:.2f}")
    print(f"  generation_ms: {generation_ms:.2f}")
    print(f"  total_ms: {retrieval_ms + generation_ms:.2f}")


if __name__ == "__main__":
    main()
