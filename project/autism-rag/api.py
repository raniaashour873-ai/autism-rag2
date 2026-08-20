"""
API Layer: wraps full_pipeline for the frontend.
Routine support is a separate endpoint and does not use medical retrieval.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import TOP_K_DEFAULT
from routine_generator import generate_routine
from step4_retrieval import get_index_stats
from step8_full_pipeline import full_pipeline

app = FastAPI(title="Autism Clinical RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    stats = get_index_stats()
    return {
        "status": "ok",
        "message": "Autism Clinical RAG API is running",
        "index": stats,
    }


@app.get("/health")
def health_alias():
    return health_check()


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return full_pipeline(request.question, top_k=request.top_k)


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
    return generate_routine(
        situation=request.situation,
        current_routine=request.current_routine,
        difficulties=request.difficulties,
        preferred_structure=request.preferred_structure,
        time_available=request.time_available,
        things_that_help=request.things_that_help,
    )
