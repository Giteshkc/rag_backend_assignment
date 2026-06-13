from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import conversation, ingestion
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.qdrant import ensure_collection_exists
from app.db.sql import create_tables

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("Starting RAG backend …")

    # Initialise infrastructure
    await create_tables()
    await ensure_collection_exists()

    logger.info("RAG backend ready.")
    yield

    logger.info("RAG backend shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Backend",
        description=(
            "Document ingestion with two chunking strategies and a "
            "conversational RAG API with Redis-backed memory and interview booking."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(ingestion.router, prefix="/api/v1")
    app.include_router(conversation.router, prefix="/api/v1")

    @app.get("/health", tags=["meta"])
    async def health_check() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
