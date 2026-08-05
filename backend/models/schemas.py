from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str = Field(..., description="Identifier of the indexed document")
    status: str = Field(..., description="Ingestion status")


class AskRequest(BaseModel):
    document_id: str
    question: str
    chat_history: list["ChatMessage"] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Source(BaseModel):
    source_id: str
    chunk_id: int
    page_number: int
    chunk_text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
