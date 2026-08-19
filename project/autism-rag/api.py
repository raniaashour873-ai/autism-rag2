"""
API Layer: بيغلف الـ full_pipeline في endpoint واحد بسيط عشان
الـ Frontend يقدر يستدعيه مباشرة من الواجهة.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from step8_full_pipeline import full_pipeline
from routine_generator import generate_routine

app = FastAPI(title="Autism Clinical RAG API")

# CORS: بيسمح للـ Frontend (اللي هيشتغل على رابط/بورت مختلف زي localhost:3000)
# إنه يقدر يكلم الـ API ده من غير ما المتصفح يمنعه لأسباب أمنية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في المشروع الحقيقي بعد الهاكاثون، تحدد الدومين الفعلي بدل *
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceItem(BaseModel):
    page: int
    section: str
    distance: float


class AnswerResponse(BaseModel):
    status: str          # "answered" أو "refused"
    safety_label: str     # "ALLOWED" / "NEEDS_CAUTION" / "REFUSE"
    answer: str
    sources: list[SourceItem]


@app.get("/")
def health_check():
    """
    endpoint بسيط للتأكد إن الـ API شغال - مفيد وقت الـ deployment
    """
    return {"status": "ok", "message": "Autism Clinical RAG API is running"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    الـ endpoint الأساسي: بياخد سؤال، بيرجع إجابة + مصادر + حالة الأمان
    """
    result = full_pipeline(request.question, top_k=request.top_k)
    return result


class RoutineRequest(BaseModel):
    situation: str

@app.post("/routine")
def routine(request: RoutineRequest):
    return generate_routine(request.situation)