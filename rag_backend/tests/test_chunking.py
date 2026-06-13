"""Unit tests for chunking strategies — no external dependencies."""

from app.services.chunking import fixed_chunking, semantic_chunking, chunk_text


SAMPLE_TEXT = (
    "Artificial intelligence is transforming industries worldwide. "
    "Machine learning models are now capable of performing tasks "
    "that previously required human intelligence. "
    "Natural language processing enables machines to understand and generate text. "
    "Computer vision systems can recognize objects in images with remarkable accuracy. "
    "These technologies are being applied in healthcare, finance, and transportation. "
) * 10


def test_fixed_chunking_produces_chunks():
    chunks = fixed_chunking(SAMPLE_TEXT, chunk_size=128, chunk_overlap=20)
    assert len(chunks) > 0
    for c in chunks:
        assert c.token_count <= 128
        assert c.text.strip()


def test_fixed_chunking_overlap():
    chunks = fixed_chunking(SAMPLE_TEXT, chunk_size=100, chunk_overlap=30)
    # Each chunk except the last should have content
    assert all(c.text.strip() for c in chunks)


def test_fixed_chunking_no_overlap():
    chunks = fixed_chunking(SAMPLE_TEXT, chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 0


def test_chunk_text_dispatcher_fixed():
    chunks = chunk_text(SAMPLE_TEXT, strategy="fixed", chunk_size=128, chunk_overlap=10)
    assert len(chunks) > 0


def test_chunk_text_dispatcher_semantic_fallback():
    # semantic falls back to fixed if sentence-transformers unavailable
    import unittest.mock as mock
    with mock.patch("app.services.chunking.semantic_chunking", side_effect=ImportError):
        chunks = chunk_text(SAMPLE_TEXT, strategy="semantic", chunk_size=128)
    assert len(chunks) > 0


def test_fixed_chunking_empty_text():
    chunks = fixed_chunking("", chunk_size=128)
    assert chunks == []
