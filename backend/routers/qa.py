import openai
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models.schemas import AskRequest, AskResponse
from services.qa_service import answer_question, stream_answer_question

router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    try:
        return answer_question(
            payload.document_id,
            payload.question,
            payload.chat_history,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc


@router.post("/ask/stream")
async def ask_question_stream(payload: AskRequest) -> StreamingResponse:
    try:
        stream = stream_answer_question(
            payload.document_id,
            payload.question,
            payload.chat_history,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
