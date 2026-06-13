# RAG Backend — Document Ingestion & Conversational AI

A production-ready FastAPI backend with document ingestion, vector search, and conversational RAG with interview booking support.

## Architecture

```
rag_backend/
├── app/
│   ├── api/routes/
│   │   ├── ingestion.py        # Document upload & chunking & embedding
│   │   └── conversation.py     # RAG chat + interview booking
│   ├── core/
│   │   ├── config.py           # Settings via pydantic-settings
│   │   └── logging.py          # Structured logging
│   ├── db/
│   │   ├── sql.py              # SQLAlchemy async engine + session
│   │   ├── qdrant.py           # Qdrant vector store client
│   │   └── redis.py            # Redis chat memory client
│   ├── models/
│   │   └── sql_models.py       # SQLAlchemy ORM models
│   ├── schemas/
│   │   ├── ingestion.py        # Pydantic request/response schemas
│   │   └── conversation.py     # Chat + booking schemas
│   ├── services/
│   │   ├── chunking.py         # Fixed-size & semantic chunking strategies
│   │   ├── embeddings.py       # OpenAI embedding generation
│   │   ├── vector_store.py     # Qdrant upsert/query wrapper
│   │   ├── memory.py           # Redis-backed chat history
│   │   ├── rag.py              # Custom RAG pipeline (no LangChain chains)
│   │   └── booking.py          # LLM-driven interview booking extraction
│   └── main.py                 # FastAPI app factory
├── tests/
│   ├── test_ingestion.py
│   └── test_conversation.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Vector Store | Qdrant |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Chat Memory | Redis |
| Metadata DB | PostgreSQL (async via SQLAlchemy + asyncpg) |
| Chunking | Fixed-size & Semantic (sentence-transformers) |

## Setup

### 1. Clone & install

```bash
git clone <repo-url>
cd rag_backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys
```

### 3. Start infrastructure

```bash
docker-compose up -d   # starts Qdrant, Redis, PostgreSQL
```

### 4. Run migrations & start server

```bash
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Reference

### Document Ingestion

#### `POST /api/v1/ingest/upload`

Upload a `.pdf` or `.txt` file and ingest it into the vector store.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | ✅ | `.pdf` or `.txt` |
| `chunking_strategy` | string | ✅ | `fixed` or `semantic` |
| `chunk_size` | int | ❌ | Tokens per chunk (default 512, fixed strategy) |
| `chunk_overlap` | int | ❌ | Overlap tokens (default 50, fixed strategy) |

**Response:**

```json
{
  "document_id": "uuid",
  "filename": "report.pdf",
  "chunking_strategy": "fixed",
  "total_chunks": 42,
  "status": "ingested"
}
```

#### `GET /api/v1/ingest/documents`

List all ingested documents with metadata.

---

### Conversational RAG

#### `POST /api/v1/chat/message`

Send a message and receive a context-aware RAG response.

**Body:**

```json
{
  "session_id": "user-session-uuid",
  "message": "What are the key findings in the uploaded report?",
  "document_id": "optional-filter-uuid"
}
```

**Response:**

```json
{
  "session_id": "user-session-uuid",
  "answer": "Based on the document...",
  "sources": [
    {
      "chunk_id": "uuid",
      "text": "...",
      "score": 0.91,
      "document_id": "uuid",
      "filename": "report.pdf"
    }
  ],
  "booking": null
}
```

#### `POST /api/v1/chat/message` — Interview Booking

When the user expresses intent to book an interview, the LLM extracts structured booking data:

**Example exchange:**

```
User: "I'd like to book an interview. I'm Jane Doe, jane@example.com, on July 15th at 2 PM."
```

**Response:**

```json
{
  "session_id": "...",
  "answer": "Your interview has been booked for July 15 at 2:00 PM. We'll send a confirmation to jane@example.com.",
  "sources": [],
  "booking": {
    "booking_id": "uuid",
    "name": "Jane Doe",
    "email": "jane@example.com",
    "date": "2025-07-15",
    "time": "14:00",
    "status": "confirmed"
  }
}
```

#### `GET /api/v1/chat/history/{session_id}`

Retrieve full conversation history for a session.

#### `DELETE /api/v1/chat/history/{session_id}`

Clear conversation history for a session.

#### `GET /api/v1/chat/bookings`

List all interview bookings.

## Chunking Strategies

### Fixed-size (`fixed`)
Splits text into chunks of `chunk_size` tokens with `chunk_overlap` token overlap. Fast and deterministic.

### Semantic (`semantic`)
Groups sentences into chunks based on embedding similarity using a sliding window. Produces semantically coherent chunks at the cost of higher ingestion time.

## Environment Variables

See `.env.example` for all required variables.
