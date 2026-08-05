# RAG-Based Document Q&A Application

A full-stack Retrieval-Augmented Generation (RAG) app where users upload a PDF, ask questions, and get answers grounded only in that document with source citations.

## Features

- PDF upload and ingestion
- Token-aware chunking (512 tokens with overlap)
- Batched OpenAI embeddings
- FAISS semantic retrieval
- Answer generation constrained to retrieved document context
- Source citations in answer output (`S# | page N`)
- Streaming responses (SSE)
- Multi-turn chat support

## Tech Stack

### Backend
- Python 3.11+
- FastAPI
- OpenAI SDK
- FAISS (`faiss-cpu`)
- PyMuPDF
- `langchain-text-splitters` (only for `RecursiveCharacterTextSplitter`)
- `tiktoken`
- Pytest

### Frontend
- React
- Vite
- Tailwind CSS

## Architecture (End to End)

1. User uploads a PDF from the React UI.
2. Backend extracts page text using PyMuPDF.
3. Text is split into token-aware chunks.
4. Chunks are embedded in batches and stored in FAISS with JSON metadata.
5. For each question, the query is embedded and top-k similar chunks are retrieved.
6. Low-score and overlapping chunks are filtered.
7. Retrieved context is sent to the chat model with strict grounding instructions.
8. Answer + citations are returned (regular or streaming).

## Project Structure

```text
rag-doc-qa/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   ├── services/
│   ├── models/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   └── .env.example
├── .env.example
└── README.md
```

## Local Setup

### 1) Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create backend `.env` from root example:

```bash
cp ../.env.example .env
```

Update at least:
- `OPENAI_API_KEY`

Run backend:

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`.

### 2) Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Frontend runs at `http://localhost:5173`.

## API Endpoints

- `POST /api/upload` - Upload PDF and build vector index
- `POST /api/ask` - Non-streaming answer
- `POST /api/ask/stream` - Streaming answer (SSE)
- `GET /api/health` - Health check

## Testing

From `backend/`:

```bash
pytest tests/unit tests/integration
```

Optional live smoke tests (requires real API key and enabled env flag):

```bash
RUN_LIVE_OPENAI_TESTS=true pytest tests/live_smoke
```
