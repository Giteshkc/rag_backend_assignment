"""
Document Ingestion API
======================
POST /api/v1/ingest/upload   – upload & ingest a .pdf or .txt file
GET  /api/v1/ingest/documents – list all ingested documents
"""

from __future__ import annotations

import io
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.sql import get_db_session
from app.models.sql_models import Chunk, Document
from app.schemas.ingestion import (
    ChunkingStrategy,
    DocumentListResponse,
    DocumentMeta,
    IngestionResponse,
)
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts
from app.services.vector_store import upsert_chunks

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


# ── Text extraction ────────────────────────────────────────────────────────────

def _extract_text_from_pdf(data: bytes) -> str:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        return _extract_text_from_pdf(data)
    # .txt
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Could not decode text file; unsupported encoding.",
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    file: UploadFile,
    chunking_strategy: ChunkingStrategy = Form(...),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    db: AsyncSession = Depends(get_db_session),
) -> IngestionResponse:
    settings = get_settings()

    # ── Validate file type ─────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"pdf", "txt"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '.{ext}'. Accepted: pdf, txt",
        )

    # ── Validate file size ─────────────────────────────────────────────────────
    raw_data = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(raw_data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    # ── Extract text ───────────────────────────────────────────────────────────
    text = _extract_text(file.filename, raw_data)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Extracted text is empty.")

    # ── Chunk ──────────────────────────────────────────────────────────────────
    chunks = chunk_text(
        text,
        strategy=chunking_strategy.value,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from document.")

    # ── Save document metadata ─────────────────────────────────────────────────
    document_id = str(uuid.uuid4())
    doc = Document(
        id=document_id,
        filename=file.filename,
        file_type=ext,
        chunking_strategy=chunking_strategy.value,
        total_chunks=len(chunks),
    )
    db.add(doc)

    # ── Save chunk metadata ────────────────────────────────────────────────────
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []

    for c in chunks:
        cid = str(uuid.uuid4())
        chunk_ids.append(cid)
        chunk_texts.append(c.text)
        db.add(
            Chunk(
                id=cid,
                document_id=document_id,
                chunk_index=c.index,
                text=c.text,
                token_count=c.token_count,
            )
        )

    # Flush to DB before touching external services so IDs are stable
    await db.flush()

    # ── Generate embeddings ────────────────────────────────────────────────────
    logger.info("Generating embeddings for %d chunks …", len(chunk_texts))
    embeddings = await embed_texts(chunk_texts)

    # ── Upsert into Qdrant ─────────────────────────────────────────────────────
    await upsert_chunks(
        document_id=document_id,
        filename=file.filename,
        chunk_ids=chunk_ids,
        texts=chunk_texts,
        embeddings=embeddings,
    )

    logger.info(
        "Ingested document '%s' | id=%s | chunks=%d | strategy=%s",
        file.filename,
        document_id,
        len(chunks),
        chunking_strategy.value,
    )

    return IngestionResponse(
        document_id=document_id,
        filename=file.filename,
        chunking_strategy=chunking_strategy,
        total_chunks=len(chunks),
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db_session),
) -> DocumentListResponse:
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return DocumentListResponse(
        documents=[DocumentMeta.model_validate(d) for d in docs],
        total=len(docs),
    )
