from pydantic import BaseModel


class ApproveRequest(BaseModel):
    reviewer: str


class EditRequest(BaseModel):
    reviewer: str
    edited_text: str


class RejectRequest(BaseModel):
    reviewer: str
    reason: str
