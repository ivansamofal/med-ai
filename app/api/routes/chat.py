import asyncio

from fastapi import APIRouter

from app.agent.service import send_chat_message
from app.api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat")
async def chat(payload: ChatRequest) -> ChatResponse:
    reply = await asyncio.to_thread(
        send_chat_message, payload.session_id, payload.patient_id, payload.message
    )
    return ChatResponse(reply=reply)
