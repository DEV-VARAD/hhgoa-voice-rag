"""Passage flattening and chunking helpers for the MSMARCO-XI RAG dataset.

The public functions operate on lightweight document dictionaries so that the
output of :func:`flatten_passages` can be passed directly to any strategy.
Each document has ``text`` and ``metadata`` keys; each chunk has the same
shape.
"""

from __future__ import annotations

import argparse
import re
import pyarrow.parquet as pq
from collections import defaultdict
from statistics import mean
from typing import Any, Callable, Iterable, Mapping


DEFAULT_CHUNK_SIZE = 256


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dict-like or attribute-style struct."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def flatten_passages(dataframe: Any) -> list[dict[str, Any]]:
    """Flatten English and Hindi passages from an MSMARCO-XI DataFrame.

    One document is emitted for every non-empty passage in
    ``passages.English_passages`` and ``passages.Translated_passages``.  The
    corresponding ``is_selected`` value is attached by list position; a
    missing flag is treated as ``0``.  This accommodates common pandas/
    PyArrow representations of a parquet struct (mapping or attribute style).
    """
    documents: list[dict[str, Any]] = []
    language_fields = (
        ("English_passages", "en"),
        ("Translated_passages", "hi"),
    )

    for _, row in dataframe.iterrows():
        passages = row["passages"]
        selected_flags = _field(passages, "is_selected", []) or []
        metadata_base = {
            "query_id": row["query_id"],
            "query_type": row["query_type"],
        }

        for passage_field, language in language_fields:
            passage_list = _field(passages, passage_field, []) or []
            for passage_index, passage in enumerate(passage_list):
                text = str(passage).strip() if passage is not None else ""
                if not text:
                    continue
                is_selected = (
                    int(selected_flags[passage_index])
                    if passage_index < len(selected_flags)
                    else 0
                )
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            **metadata_base,
                            "language": language,
                            "is_selected": is_selected,
                        },
                    }
                )
    return documents


def _word_windows(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    """Yield overlapping whitespace-word windows from text."""
    words = text.split()
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            yield " ".join(window)
        if start + chunk_size >= len(words):
            break


def fixed_size_chunking(
    docs: list[dict[str, Any]], chunk_size: int = 256, overlap: int = 50
) -> list[dict[str, Any]]:
    """Chunk documents into fixed, overlapping whitespace-word windows.

    This is fast, deterministic, and preserves retrieval recall around window
    boundaries through overlap.  Its tradeoff is that it can cut through a
    sentence or idea, making chunks less coherent than semantic chunking.
    ``chunk_size`` and ``overlap`` are words, not model tokens.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    for document in docs:
        for chunk_text in _word_windows(document["text"], chunk_size, overlap):
            chunks.append({"text": chunk_text, "metadata": dict(document["metadata"])})
    return chunks


def _regex_sentences(text: str) -> list[str]:
    """Split Hindi-friendly text after danda and ordinary sentence markers."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[।॥.!?])\s+|(?<=[।॥.!?])$", text.strip())
        if sentence.strip()
    ]


def _english_sentences(text: str) -> list[str]:
    """Use nltk's English splitter, with a regex fallback if punkt is absent."""
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text, language="english")
    except (ImportError, LookupError):
        return _regex_sentences(text)


def _sentences_for(document: Mapping[str, Any]) -> list[str]:
    language = str(document.get("metadata", {}).get("language", "")).lower()
    if language == "en":
        return _english_sentences(document["text"])
    if language == "hi":
        return _regex_sentences(document["text"])
    # An unknown language may still be English; use nltk as the conservative fallback.
    return _english_sentences(document["text"])


def _split_oversized_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Last recursive level: split a sentence that exceeds the word budget."""
    return list(_word_windows(sentence, chunk_size, overlap=0))


def semantic_chunking(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build sentence-aware chunks, targeting up to 256 whitespace words.

    The splitter recursively falls back from a document to sentences and then
    to word windows only when an individual sentence is too long.  That keeps
    complete sentences together whenever possible, at the cost of less exact
    and sometimes uneven chunk sizes than fixed windows.

    English documents use ``nltk.sent_tokenize``. Hindi deliberately uses a
    regex that recognizes Devanagari danda (``।``), double danda (``॥``), and
    ``.``, ``!``, and ``?``; nltk's sentence tokenizer alone is not reliable
    for those Hindi markers. If ``metadata.language`` is ambiguous, the code
    falls back to nltk's English-oriented splitter. If NLTK's punkt data is
    unavailable, a punctuation regex is used rather than failing at runtime.
    """
    chunks: list[dict[str, Any]] = []
    for document in docs:
        current: list[str] = []
        current_words = 0
        sentences = _sentences_for(document) or [document["text"]]

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if sentence_words > DEFAULT_CHUNK_SIZE:
                if current:
                    chunks.append({"text": " ".join(current), "metadata": dict(document["metadata"])})
                    current, current_words = [], 0
                for piece in _split_oversized_sentence(sentence, DEFAULT_CHUNK_SIZE):
                    chunks.append({"text": piece, "metadata": dict(document["metadata"])})
            elif current and current_words + sentence_words > DEFAULT_CHUNK_SIZE:
                chunks.append({"text": " ".join(current), "metadata": dict(document["metadata"])})
                current, current_words = [sentence], sentence_words
            else:
                current.append(sentence)
                current_words += sentence_words

        if current:
            chunks.append({"text": " ".join(current), "metadata": dict(document["metadata"])})
    return chunks


def metadata_aware_chunking(
    docs: list[dict[str, Any]], chunk_size: int = 256
) -> list[dict[str, Any]]:
    """Use non-overlapping fixed word windows while recording chunk provenance.

    This mirrors fixed-size chunking without overlap and adds the source
    ``query_id``, a zero-based ``chunk_index`` within the source document,
    ``language``, and ``is_selected`` to each metadata object. It is useful
    when retrieval results must be traced or grouped back to source passages;
    the extra metadata slightly increases index/storage size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks: list[dict[str, Any]] = []
    for document in docs:
        source_metadata = document["metadata"]
        for chunk_index, chunk_text in enumerate(
            _word_windows(document["text"], chunk_size, overlap=0)
        ):
            metadata = dict(source_metadata)
            metadata.update(
                {
                    "source_query_id": source_metadata.get("query_id"),
                    "chunk_index": chunk_index,
                    "language": source_metadata.get("language"),
                    "is_selected": source_metadata.get("is_selected"),
                }
            )
            chunks.append({"text": chunk_text, "metadata": metadata})
    return chunks


def _sample_documents(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select up to 25 English and 25 Hindi documents for comparison."""
    english = [doc for doc in docs if doc["metadata"].get("language") == "en"][:25]
    hindi = [doc for doc in docs if doc["metadata"].get("language") == "hi"][:25]
    return english + hindi


def _print_stats(strategy_name: str, chunks: list[dict[str, Any]]) -> None:
    """Print word-length statistics per language for one chunking strategy."""
    by_language: dict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        language = chunk["metadata"].get("language", "unknown")
        by_language[language].append(len(chunk["text"].split()))

    print(f"\n{strategy_name}")
    for language in ("en", "hi"):
        lengths = by_language.get(language, [])
        if lengths:
            print(
                f"  {language}: chunks={len(lengths)}, avg_words={mean(lengths):.1f}, "
                f"min_words={min(lengths)}, max_words={max(lengths)}"
            )
        else:
            print(f"  {language}: no chunks")


def main() -> None:
    """Run a 25-English/25-Hindi chunking comparison from a parquet file."""
    parser = argparse.ArgumentParser(description="Compare MSMARCO-XI chunking strategies.")
    parser.add_argument("parquet_path", help="Path to the MSMARCO-XI Hindi parquet file")
    args = parser.parse_args()

    import pandas as pd
    import duckdb
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{args.parquet_path}') LIMIT 20000").df()
    documents = flatten_passages(df)
    sample = _sample_documents(documents)
    print(f"Comparing {len(sample)} documents (up to 25 English + 25 Hindi).")
    strategies: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "Fixed-size chunking": fixed_size_chunking,
    "Semantic chunking": semantic_chunking,
    "Metadata-aware chunking": metadata_aware_chunking,
}
    for name, strategy in strategies.items():
        _print_stats(name, strategy(sample))


if __name__ == "__main__":
    main()
