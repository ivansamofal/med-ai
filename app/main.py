from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.lab_results import router as lab_results_router
from app.config import settings
from app.db.mongo import close_client
from app.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    yield
    close_client()


app = FastAPI(title="MedAI", lifespan=lifespan)
app.include_router(lab_results_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
