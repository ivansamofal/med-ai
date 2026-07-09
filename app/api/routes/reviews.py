import asyncio

from fastapi import APIRouter

from app.api.schemas.review import ApproveRequest, EditRequest, RejectRequest
from app.approval.service import resume_review

router = APIRouter()


@router.post("/reviews/{recommendation_id}/approve")
async def approve_review(recommendation_id: str, payload: ApproveRequest) -> dict[str, str]:
    await asyncio.to_thread(resume_review, recommendation_id, "approved", payload.reviewer)
    return {"status": "approved"}


@router.post("/reviews/{recommendation_id}/edit")
async def edit_review(recommendation_id: str, payload: EditRequest) -> dict[str, str]:
    await asyncio.to_thread(
        resume_review, recommendation_id, "edited", payload.reviewer, payload.edited_text
    )
    return {"status": "edited"}


@router.post("/reviews/{recommendation_id}/reject")
async def reject_review(recommendation_id: str, payload: RejectRequest) -> dict[str, str]:
    await asyncio.to_thread(
        resume_review, recommendation_id, "rejected", payload.reviewer, None, payload.reason
    )
    return {"status": "rejected"}
