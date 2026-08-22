"""Guarded end-to-end orchestration for the multilingual RAG pipeline.

This module loads the already-built index once, applies inexpensive input and
relevance checks, then invokes retrieval and grounded Groq generation only
when appropriate. Every processed query is emitted as a JSON log line for
later latency analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from embed_index import load_index
from generation import DEFAULT_GROQ_MODEL, generate_answer
from retrieval import retrieve


LOGGER_NAME = "rag_harness"
LOG_FILE = "harness.log"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
DEFAULT_SCORE_THRESHOLD = 0.3
INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough information to answer that"
UNSAFE_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "you are now",
)


def _configure_logger() -> logging.Logger:
    """Create one stdout/file logger without duplicating handlers on re-import."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger


LOGGER = _configure_logger()


def _log_request(query: str, status: str, latency_ms: dict, **extra: Any) -> None:
    """Emit one JSON line per request to both configured logging destinations."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query[:100],
        "status": status,
        "latency_ms": {k: round(float(v), 3) for k, v in latency_ms.items()},
        **extra,
    }
    LOGGER.info(json.dumps(event, ensure_ascii=False, default=str))


def _guardrail_reason(query: str) -> str | None:
    """Return a rejection reason for empty, malformed, or obvious injection input."""
    if not isinstance(query, str) or not query.strip():
        return "empty_query"
    normalized_query = query.strip()
    if len(normalized_query) < 3:
        return "query_too_short"
    if len(normalized_query) > 500:
        return "query_too_long"
    lowered_query = normalized_query.lower()
    if any(pattern in lowered_query for pattern in UNSAFE_PATTERNS):
        return "unsafe_prompt_injection_pattern"
    return None


def process_query(
    query: str,
    faiss_index: Any,
    metadata_list: list[dict],
    embed_model: Any,
    groq_api_key: str,
    groq_model: str,
    top_k: int = 5,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> dict:
    """Run guarded retrieval and generation for one user query.

    Guardrails reject malformed or obvious prompt-injection input before any
    model work. Retrieval and generation failures are contained and converted
    into structured result dictionaries, so callers never need to handle a
    pipeline crash for an individual request.
    """
    request_started = time.perf_counter()
    guardrail_reason = _guardrail_reason(query)
    if guardrail_reason:
        result = {"status": "rejected", "reason": guardrail_reason, "answer": None}
        _log_request(query if isinstance(query, str) else "", result["status"], {"total": 0.0})
        return result
    if not 0.0 <= score_threshold <= 1.0:
        result = {"status": "error", "reason": "invalid_score_threshold", "answer": None}
        _log_request(query, result["status"], {"total": (time.perf_counter() - request_started) * 1000})
        return result

    retrieval_started = time.perf_counter()
    try:
        retrieved_chunks = retrieve(query, faiss_index, metadata_list, embed_model, top_k)
    except Exception as error:  # Retrieval must not terminate the harness.
        total_ms = (time.perf_counter() - request_started) * 1000
        result = {"status": "error", "reason": "retrieval_failed", "answer": None}
        _log_request(query, result["status"], {"total": total_ms}, error=str(error))
        return result
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    top_score = float(retrieved_chunks[0].get("score", 0.0)) if retrieved_chunks else 0.0
    if top_score < score_threshold:
        total_ms = (time.perf_counter() - request_started) * 1000
        result = {
            "status": "no_relevant_context",
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "sources": [],
        }
        _log_request(query, result["status"], {"retrieval": retrieval_ms, "total": total_ms})
        return result

    try:
        generated = generate_answer(query, retrieved_chunks, groq_api_key, groq_model)
    except Exception as error:  # Generation already retries transient Groq failures.
        total_ms = (time.perf_counter() - request_started) * 1000
        result = {"status": "error", "reason": "generation_failed", "answer": None}
        _log_request(query, result["status"], {"retrieval": retrieval_ms, "total": total_ms}, error=str(error))
        return result

    generation_ms = float(generated["latency_ms"])
    total_ms = retrieval_ms + generation_ms
    result = {
        "status": "success",
        "answer": str(generated["answer"]),
        "sources": [
            {
                "text": chunk["text"],
                "language": chunk.get("metadata", {}).get("language", "unknown"),
                "score": float(chunk["score"]),
            }
            for chunk in retrieved_chunks
        ],
        "latency_ms": {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "total": total_ms,
        },
        "model_used": str(generated["model_used"]),
    }
    _log_request(query, result["status"], result["latency_ms"])
    return result


def _percentile(latencies: list[float], percentile: int) -> float:
    """Calculate a nearest-rank percentile, suitable for small benchmark runs."""
    if not latencies:
        return 0.0
    ordered = sorted(latencies)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[index]


def _print_benchmark_summary(latencies: list[float]) -> None:
    """Print copy-ready P50/P70/P100 total-latency metrics for benchmark runs."""
    print("\nBenchmark latency report (total_ms across all runs):")
    print(f"  Runs: {len(latencies)}")
    print(f"  P50: {_percentile(latencies, 50):.2f} ms")
    print(f"  P70: {_percentile(latencies, 70):.2f} ms")
    print(f"  P100: {_percentile(latencies, 100):.2f} ms")


def main() -> None:
    """Run one query or a benchmark file through the complete RAG pipeline."""
    parser = argparse.ArgumentParser(description="Run the guarded multilingual RAG pipeline.")
    parser.add_argument("--index_path", default="index.faiss")
    parser.add_argument("--metadata_path", default="metadata.json")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Embedding model label; the saved index configuration chooses the loaded model.",
    )
    parser.add_argument("--groq_model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--groq_api_key", default=os.environ.get("GROQ_API_KEY"))
    parser.add_argument("--query", help="Single query to process")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--score_threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_queries_file", help="One benchmark query per line")
    args = parser.parse_args()

    if args.benchmark and not args.benchmark_queries_file:
        parser.error("--benchmark requires --benchmark_queries_file")
    if not args.benchmark and args.query is None:
        parser.error("--query is required unless --benchmark is used")

    try:
        faiss_index, metadata_list, embed_model = load_index(
            args.index_path, args.metadata_path
        )
        if args.benchmark:
            queries = Path(args.benchmark_queries_file).read_text(encoding="utf-8").splitlines()
            benchmark_latencies: list[float] = []
            for query in queries:
                run_started = time.perf_counter()
                process_query(
                    query,
                    faiss_index,
                    metadata_list,
                    embed_model,
                    args.groq_api_key,
                    args.groq_model,
                    args.top_k,
                    args.score_threshold,
                )
                benchmark_latencies.append((time.perf_counter() - run_started) * 1000)
                time.sleep(3)
            _print_benchmark_summary(benchmark_latencies)
        else:
            result = process_query(
                args.query,
                faiss_index,
                metadata_list,
                embed_model,
                args.groq_api_key,
                args.groq_model,
                args.top_k,
                args.score_threshold,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (FileNotFoundError, ImportError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
