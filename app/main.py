from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.lab_results import router as lab_results_router
from app.api.routes.reviews import router as reviews_router
from app.config import settings
from app.db.mongo import close_client
from app.db.sync_mongo import close_sync_client
from app.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    yield
    close_client()
    close_sync_client()


app = FastAPI(title="MedAI", lifespan=lifespan)
app.include_router(lab_results_router)
app.include_router(reviews_router)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
