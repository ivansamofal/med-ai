from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    patient_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
