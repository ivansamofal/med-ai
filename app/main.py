from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.appointments import router as appointments_router
from app.api.routes.audit_log import router as audit_log_router
from app.api.routes.chat import router as chat_router
from app.api.routes.lab_results import router as lab_results_router
from app.api.routes.recommendations import router as recommendations_router
from app.api.routes.reviews import router as reviews_router
from app.config import settings
from app.db.mongo import close_client
from app.db.sync_mongo import close_sync_client
from app.logging import configure_logging

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


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
app.include_router(recommendations_router)
app.include_router(appointments_router)
app.include_router(audit_log_router)
# Read-only dashboard (static HTML/JS, no build step) over the read
# endpoints above — not part of the phased project spec, added on request
# to have something to look at while seeding/demoing.
app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
