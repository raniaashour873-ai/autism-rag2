"""
API Layer: wraps full_pipeline for the frontend.
Routine support is a separate endpoint and does not use medical retrieval.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import TOP_K_DEFAULT, cors_allow_origins, process_rss_mb
from routine_generator import generate_routine
from step8_full_pipeline import full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("autism_rag.api")

app = FastAPI(title="Autism Clinical RAG API")

_FRONTEND_INDEX = Path(__file__).resolve().parents[2] / "frontend" / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)


class SourceItem(BaseModel):
    chunk_id: str = ""
    document: str = ""
    source_url: str = ""
    section: str = ""
    page: int = 0
    distance: float = 0.0
    text: str = ""


class AnswerResponse(BaseModel):
    status: str
    safety_label: str
    refuse_reason: str | None = None
    answer: str
    recommendation: str = ""
    evidence: str = ""
    citations: list[SourceItem] = []
    sources: list[SourceItem]
    generation_called: bool = False


class RoutineRequest(BaseModel):
    situation: str = ""
    current_routine: str = ""
    difficulties: str = ""
    preferred_structure: str = ""
    time_available: str = ""
    things_that_help: str = ""


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Autism Clinical RAG API is running",
        "index": {"chunk_count": None, "documents": []},
    }


@app.get("/health")
def health_alias():
    return {"status": "ok"}


@app.get("/ui")
def serve_ui():
    if not _FRONTEND_INDEX.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(_FRONTEND_INDEX, media_type="text/html")


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        logger.info("/ask start (rss_mb=%s)", process_rss_mb())
        result = full_pipeline(request.question, top_k=request.top_k)
        logger.info(
            "/ask done status=%s refuse=%s gen=%s (rss_mb=%s)",
            result.get("status"),
            result.get("refuse_reason"),
            result.get("generation_called"),
            process_rss_mb(),
        )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("/ask failed")
        raise HTTPException(
            status_code=503,
            detail="The assistant could not complete this request. Please retry.",
        )


@app.post("/routine")
def routine(request: RoutineRequest):
    blob = " ".join(
        [
            request.situation,
            request.current_routine,
            request.difficulties,
            request.preferred_structure,
            request.time_available,
            request.things_that_help,
        ]
    )
    if not blob.strip():
        raise HTTPException(status_code=400, detail="routine description must not be empty")
    try:
        return generate_routine(
            situation=request.situation,
            current_routine=request.current_routine,
            difficulties=request.difficulties,
            preferred_structure=request.preferred_structure,
            time_available=request.time_available,
            things_that_help=request.things_that_help,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("/routine failed")
        raise HTTPException(
            status_code=503,
            detail="The routine helper could not complete this request. Please retry.",
        )
