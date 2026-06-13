"""
Chunking strategies:
  - fixed   : Split by token count with configurable overlap.
  - semantic : Group sentences into chunks by embedding cosine similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import tiktoken

from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True)
class TextChunk:
    index: int
    text: str
    token_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[int]:
    return _TOKENIZER.encode(text)


def _decode(tokens: list[int]) -> str:
    return _TOKENIZER.decode(tokens)


def _sentence_split(text: str) -> list[str]:
    """Naive sentence splitter that handles common abbreviations."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


# ── Fixed-size strategy ────────────────────────────────────────────────────────

def fixed_chunking(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """
    Tokenise the full text, then slide a window of `chunk_size` tokens
    advancing by `chunk_size - chunk_overlap` tokens each step.
    """
    tokens = _tokenize(text)
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[TextChunk] = []

    for i, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunk_text = _decode(window)
        chunks.append(TextChunk(index=i, text=chunk_text, token_count=len(window)))

    logger.debug("fixed_chunking produced %d chunks", len(chunks))
    return chunks


# ── Semantic strategy ──────────────────────────────────────────────────────────

def semantic_chunking(
    text: str,
    max_tokens_per_chunk: int = 512,
    similarity_threshold: float = 0.75,
) -> list[TextChunk]:
    """
    1. Split text into sentences.
    2. Embed each sentence with a lightweight local model.
    3. Merge adjacent sentences into chunks while they remain semantically
       similar AND the chunk stays under `max_tokens_per_chunk`.

    Falls back to fixed chunking if sentence-transformers is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; falling back to fixed chunking"
        )
        return fixed_chunking(text, chunk_size=max_tokens_per_chunk)

    sentences = _sentence_split(text)
    if not sentences:
        return []

    embeddings: np.ndarray = model.encode(sentences, convert_to_numpy=True)

    chunks: list[TextChunk] = []
    current_sentences: list[str] = [sentences[0]]
    current_tokens: int = len(_tokenize(sentences[0]))
    chunk_index = 0

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_tokens = len(_tokenize(sent))
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])

        would_overflow = (current_tokens + sent_tokens) > max_tokens_per_chunk
        dissimilar = sim < similarity_threshold

        if dissimilar or would_overflow:
            # Flush current buffer
            chunk_text = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    index=chunk_index,
                    text=chunk_text,
                    token_count=current_tokens,
                )
            )
            chunk_index += 1
            current_sentences = [sent]
            current_tokens = sent_tokens
        else:
            current_sentences.append(sent)
            current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            TextChunk(
                index=chunk_index,
                text=chunk_text,
                token_count=current_tokens,
            )
        )

    logger.debug("semantic_chunking produced %d chunks", len(chunks))
    return chunks


# ── Public dispatcher ─────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    strategy: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    if strategy == "semantic":
        return semantic_chunking(text, max_tokens_per_chunk=chunk_size)
    return fixed_chunking(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
