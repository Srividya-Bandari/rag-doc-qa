import uuid

import fitz
import openai
from fastapi import APIRouter, File, HTTPException, UploadFile

from config import get_settings
from models.schemas import UploadResponse
from services.chunker import chunk_pages
from services.embedder import embed_chunks
from services.pdf_parser import extract_text_by_page
from services.vector_store import save_document_index

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    max_upload_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(pdf_bytes) > max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {settings.max_upload_size_mb} MB upload limit.",
        )

    try:
        pages = extract_text_by_page(pdf_bytes)
    except (fitz.FileDataError, fitz.EmptyFileError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or unreadable PDF file.") from exc
    if not pages:
        raise HTTPException(status_code=422, detail="No extractable text found in this PDF.")

    document_id = str(uuid.uuid4())
    chunks = chunk_pages(pages, document_id)
    if not chunks:
        raise HTTPException(status_code=422, detail="No indexable text chunks were created from this PDF.")
    try:
        vectors = embed_chunks(chunks)
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc
    save_document_index(document_id, vectors, chunks)

    return UploadResponse(
        document_id=document_id,
        status=f"indexed:{len(chunks)}_chunks",
    )
