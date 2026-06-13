"""
Custom RAG pipeline — no LangChain RetrievalQAChain.

Flow:
  1. Load chat history from Redis
  2. Embed the user query
  3. Retrieve top-k relevant chunks from Qdrant
  4. Build a grounded prompt with history + context
  5. Call OpenAI chat completion
  6. Persist both turns to Redis
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.conversation import ChatMessage, SourceChunk
from app.services.embeddings import embed_query
from app.services.memory import append_message, get_history
from app.services.vector_store import SearchResult, similarity_search

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions strictly based on the
provided context passages.  If the context does not contain enough information
to answer confidently, say so honestly rather than speculating.

Guidelines:
- Be concise and precise.
- Cite which passage(s) support your answer when relevant.
- Maintain a professional and helpful tone.
"""


def _build_context_block(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(f"[{i}] (from '{r.filename}', score={r.score:.3f})\n{r.text}")
    return "\n\n".join(parts)


def _build_messages(
    history: list[ChatMessage],
    context_block: str,
    user_message: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Include prior turns
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})

    # Augment the current user message with retrieved context
    augmented = (
        f"Context passages:\n{context_block}\n\n---\n\nQuestion: {user_message}"
        if context_block
        else user_message
    )
    messages.append({"role": "user", "content": augmented})
    return messages


async def run_rag(
    session_id: str,
    user_message: str,
    document_id: str | None = None,
    top_k: int = 5,
) -> tuple[str, list[SourceChunk]]:
    """
    Execute the full RAG pipeline for one turn.

    Returns:
        (answer_text, source_chunks)
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # 1. Load history
    history = await get_history(session_id)

    # 2. Embed query
    query_vector = await embed_query(user_message)

    # 3. Retrieve context
    search_results = await similarity_search(
        query_vector=query_vector,
        top_k=top_k,
        document_id=document_id,
    )

    # 4. Build prompt
    context_block = _build_context_block(search_results)
    messages = _build_messages(history, context_block, user_message)

    # 5. LLM call
    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.2,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content or ""

    # 6. Persist turns
    await append_message(session_id, "user", user_message)
    await append_message(session_id, "assistant", answer)

    source_chunks = [
        SourceChunk(
            chunk_id=r.chunk_id,
            text=r.text[:300],  # truncate for response payload
            score=r.score,
            document_id=r.document_id,
            filename=r.filename,
        )
        for r in search_results
    ]

    logger.info(
        "RAG completed | session=%s | sources=%d", session_id, len(source_chunks)
    )
    return answer, source_chunks
