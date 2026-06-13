from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SEMANTIC = "semantic"


class IngestionRequest(BaseModel):
    chunking_strategy: ChunkingStrategy
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=512)


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: ChunkingStrategy
    total_chunks: int
    status: Literal["ingested"] = "ingested"


class DocumentMeta(BaseModel):
    id: str
    filename: str
    file_type: str
    chunking_strategy: str
    total_chunks: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total: int
