"""Embed RAG chunks and persist a FAISS similarity index with provenance.

The module consumes the ``{"text": ..., "metadata": ...}`` chunks produced
by :mod:`chunking`.  FAISS stores vectors only, so ``metadata.json`` preserves
the complete chunk for every vector row in the same order.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INDEX_PATH = "index.faiss"
DEFAULT_METADATA_PATH = "metadata.json"
CONFIG_FILENAME = "index_config.json"


def _dependencies() -> tuple[Any, Any, Any]:
    """Import optional embedding dependencies only when index work is requested."""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "embed_index.py requires faiss-cpu, sentence-transformers, and numpy. "
            "Install them in the active environment before building an index."
        ) from error
    return faiss, np, SentenceTransformer


def _json_default(value: Any) -> Any:
    """Serialize numpy scalar metadata values commonly produced by pandas."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _config_path(index_path: str) -> Path:
    """Keep model/index configuration next to the FAISS index file."""
    return Path(index_path).resolve().parent / CONFIG_FILENAME


def build_index(chunks: list[dict], model_name: str, index_type: str) -> None:
    """Embed chunks and save ``index.faiss`` plus its aligned metadata store.

    Embeddings are L2-normalized before insertion, so an ``IndexFlatIP``
    measures cosine similarity with exact inner-product search. ``hnsw`` uses
    ``IndexHNSWFlat`` with the same inner-product metric for faster approximate
    search at larger scale. Files are written to the current working directory
    as ``index.faiss``, ``metadata.json``, and ``index_config.json``.

    Args:
        chunks: Chunk dictionaries containing ``text`` and ``metadata`` keys.
        model_name: Any SentenceTransformer-compatible model identifier.
        index_type: ``"flat"`` for exact search or ``"hnsw"`` for approximate
            HNSW search.
    """
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list")
    if index_type not in {"flat", "hnsw"}:
        raise ValueError("index_type must be either 'flat' or 'hnsw'")

    texts: list[str] = []
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or "text" not in chunk or "metadata" not in chunk:
            raise ValueError(
                f"Chunk at position {position} must contain 'text' and 'metadata' keys"
            )
        texts.append(str(chunk["text"]))

    faiss, np, SentenceTransformer = _dependencies()
    model = SentenceTransformer(model_name)

    embedding_started = time.perf_counter()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    embedding_seconds = time.perf_counter() - embedding_started

    dimension = int(embeddings.shape[1])
    index_started = time.perf_counter()
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    else:
        # Inner product on normalized vectors is cosine similarity.
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
    index.add(embeddings)
    index_seconds = time.perf_counter() - index_started

    index_path = Path(DEFAULT_INDEX_PATH)
    metadata_path = Path(DEFAULT_METADATA_PATH)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(chunks, metadata_file, ensure_ascii=False, default=_json_default)
    with _config_path(str(index_path)).open("w", encoding="utf-8") as config_file:
        json.dump({"model_name": model_name, "index_type": index_type}, config_file)

    print("Embedding and indexing complete")
    print(f"Total documents embedded: {len(chunks)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Embedding time: {embedding_seconds:.3f} seconds")
    print(f"FAISS index build time: {index_seconds:.3f} seconds")
    print(f"Saved FAISS index: {index_path.resolve()}")
    print(f"Saved metadata store: {metadata_path.resolve()}")


def load_index(index_path: str, metadata_path: str) -> tuple[Any, list[dict], Any]:
    """Load a persisted index, its row-aligned chunk list, and its embedding model.

    The model name is read from the ``index_config.json`` stored alongside the
    index by :func:`build_index`. This lets a retrieval module call this
    function without separately repeating the model choice.
    """
    faiss, _, SentenceTransformer = _dependencies()
    index_file = Path(index_path)
    metadata_file = Path(metadata_path)
    config_file = _config_path(str(index_file))
    if not index_file.is_file():
        raise FileNotFoundError(f"FAISS index file not found: {index_file}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not config_file.is_file():
        raise FileNotFoundError(
            f"Index configuration not found: {config_file}. Rebuild the index with build_index."
        )

    with metadata_file.open(encoding="utf-8") as handle:
        metadata_list = json.load(handle)
    with config_file.open(encoding="utf-8") as handle:
        model_name = json.load(handle)["model_name"]
    faiss_index = faiss.read_index(str(index_file))
    if faiss_index.ntotal != len(metadata_list):
        raise ValueError(
            "Index/vector count does not match metadata rows; use files from the same build."
        )
    return faiss_index, metadata_list, SentenceTransformer(model_name)


def main() -> None:
    """Chunk a parquet dataset with :mod:`chunking`, then build its FAISS index."""
    parser = argparse.ArgumentParser(description="Build a multilingual FAISS RAG index.")
    parser.add_argument("--parquet_path", required=True, help="Input parquet dataset path")
    parser.add_argument(
        "--chunking_strategy",
        choices=("fixed", "semantic", "metadata_aware"),
        default="fixed",
        help="Chunking strategy imported from chunking.py (default: fixed)",
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--index_type", choices=("flat", "hnsw"), default="flat")
    args = parser.parse_args()

    import pandas as pd
    from chunking import (
        fixed_size_chunking,
        flatten_passages,
        metadata_aware_chunking,
        semantic_chunking,
    )

    import duckdb
    con = duckdb.connect()
    dataframe = con.execute(f"SELECT * FROM read_parquet('{args.parquet_path}') LIMIT 500").df()
    documents = flatten_passages(dataframe)
    strategies = {
        "fixed": fixed_size_chunking,
        "semantic": semantic_chunking,
        "metadata_aware": metadata_aware_chunking,
    }
    chunks = strategies[args.chunking_strategy](documents)
    build_index(chunks, args.model_name, args.index_type)


if __name__ == "__main__":
    main()
