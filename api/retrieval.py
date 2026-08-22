"""Query a persisted FAISS RAG index built by :mod:`embed_index`.

The query embedding is L2-normalized before inner-product search, matching the
normalization used during indexing. On normalized vectors, inner product is
cosine similarity.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import faiss
import numpy as np

from embed_index import load_index


def retrieve(
    query: str,
    faiss_index: Any,
    metadata_list: list[dict],
    model: Any,
    top_k: int = 5,
) -> list[dict]:
    """Return the most similar indexed chunks for a query.

    The query is encoded with the model used to build the index and L2
    normalized before FAISS inner-product search. Scores are cosine
    similarities clamped to the requested 0--1 display range. The metadata
    list must be in the same vector-row order as the FAISS index.

    Args:
        query: User question or search phrase.
        faiss_index: A loaded FAISS index from :func:`embed_index.load_index`.
        metadata_list: Full chunk dictionaries aligned with index row IDs.
        model: The loaded SentenceTransformer model.
        top_k: Maximum number of results to return.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if faiss_index is None or faiss_index.ntotal == 0:
        raise ValueError("The FAISS index is empty. Build an index before retrieving.")
    if not metadata_list:
        raise ValueError("The metadata store is empty. Build an index before retrieving.")
    if faiss_index.ntotal != len(metadata_list):
        raise ValueError("FAISS index rows and metadata rows do not match.")

    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)

    search_k = min(top_k, faiss_index.ntotal)
    distances, row_indices = faiss_index.search(query_embedding, search_k)
    results: list[dict] = []
    for score, row_index in zip(distances[0], row_indices[0]):
        if row_index < 0:  # FAISS uses -1 for an unavailable neighbor.
            continue
        chunk = metadata_list[int(row_index)]
        results.append(
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(np.clip(score, 0.0, 1.0)),
            }
        )
    return results


def _print_results(results: list[dict]) -> None:
    """Print retrieval results in a compact, submission-friendly format."""
    if not results:
        print("No retrieval results found.")
        return
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        preview = " ".join(result["text"].split())
        if len(preview) > 150:
            preview = f"{preview[:147]}..."
        print(
            f"{rank}. score={result['score']:.4f} "
            f"language={metadata.get('language', 'unknown')} "
            f"query_id={metadata.get('query_id', 'unknown')}\n"
            f"   {preview}"
        )


def main() -> None:
    """Load an index, retrieve matching chunks, and print latency and results."""
    parser = argparse.ArgumentParser(description="Retrieve chunks from a FAISS RAG index.")
    parser.add_argument("--index_path", default="index.faiss")
    parser.add_argument("--metadata_path", default="metadata.json")
    parser.add_argument(
        "--model_name",
        default=None,
        help=(
            "Optional expected model name. The saved index configuration selects "
            "the actual model used for retrieval."
        ),
    )
    parser.add_argument("--query", required=True, help="Query to retrieve against")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    try:
        faiss_index, metadata_list, model = load_index(args.index_path, args.metadata_path)
        retrieval_started = time.perf_counter()
        results = retrieve(args.query, faiss_index, metadata_list, model, args.top_k)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    elapsed_ms = (time.perf_counter() - retrieval_started) * 1000
    print(f"Query embedding + FAISS search time: {elapsed_ms:.2f} ms")
    _print_results(results)


if __name__ == "__main__":
    main()
