"""
Step 7: Input Risk Classification (Safety Layer)
====================================================
الهدف: قبل ما أي سؤال يوصل لطبقة البحث، نصنفه لواحدة من 3 حالات:

1. ALLOWED       - سؤال عادي عن المحتوى الطبي العام في الدليل (يكمل عادي)
2. NEEDS_CAUTION - سؤال شخصي عن حالة فرد بعينه (نكمل بس مع تحذير واضح إنه مش تشخيص)
3. REFUSE        - سؤال طارئ أو خطير أو خارج النطاق تمامًا (نوقف فورًا، مش هيدخل البحث خالص)

ده بيحمي النظام من إنه "يحاول يساعد" في مواقف مينفعش فيها LLM يتكلم أصلاً.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


CLASSIFICATION_PROMPT = """You are a safety classifier for a clinical guideline
assistant. Classify the following user question into exactly ONE category:

- ALLOWED: A general question about autism diagnosis, management, or care
  practices as covered in clinical guidelines (e.g. "What interventions are
  recommended for autism?").

- NEEDS_CAUTION: A question that asks about a specific individual's personal
  situation or symptoms (e.g. "My son shows these signs, does he have
  autism?", "Should I give my child this medication?"). These can still be
  answered with general guideline information, but must include a clear
  disclaimer that this is not an individual diagnosis.

- REFUSE: A question involving a medical emergency, immediate danger to
  someone's safety, a request for a specific drug dosage for an individual,
  or a topic completely unrelated to autism/clinical guidelines.

Respond with ONLY one word: ALLOWED, NEEDS_CAUTION, or REFUSE. Nothing else.
"""

# قائمة كلمات مفتاحية للطوارئ - بنتحقق منها فورًا وبسرعة قبل حتى ما نستخدم الـ LLM
# (أسرع وأضمن من الاعتماد على الموديل وحده في حالات حرجة زي دي)
EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "self-harm", "harm myself",
    "overdose", "emergency", "dying", "chest pain", "can't breathe",
]


def classify_query(question: str) -> str:
    """
    بترجع: 'ALLOWED' أو 'NEEDS_CAUTION' أو 'REFUSE'
    """
    # 1) فحص سريع للكلمات الخطرة - أولوية قصوى، من غير ما ننتظر رد الـ LLM
    lowered = question.lower()
    if any(keyword in lowered for keyword in EMERGENCY_KEYWORDS):
        return "REFUSE"

    # 2) تصنيف بالـ LLM لباقي الحالات
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    label = response.choices[0].message.content.strip().upper()

    # حماية إضافية: لو الموديل رجع حاجة غير متوقعة، نتعامل معاها كـ NEEDS_CAUTION
    # (أأمن اختيار افتراضي - نكمل بحذر بدل ما نرفض أو نكمل عادي من غير تحذير)
    if label not in ["ALLOWED", "NEEDS_CAUTION", "REFUSE"]:
        label = "NEEDS_CAUTION"

    return label


def contains_emergency_keywords(question: str) -> bool:
    """
    فحص مستقل يستخدم في اختيار نص الرسالة المناسب - نفس منطق الفحص
    المستخدم أصلاً في classify_query، لكن كدالة منفصلة قابلة لإعادة الاستخدام.
    """
    lowered = question.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


# Keyword lists used by classify_query_detailed / classify_population.
# EMERGENCY_KEYWORDS above is unchanged; these are non-emergency gates.
OUT_OF_SCOPE_KEYWORDS = [
    "pizza", "best topping", "stock market", "football score",
    " in cats", " in dogs", " in hamsters", "veterinary",
    "for cats", "for dogs",
]

CHILD_POPULATION_HINTS = [
    "my child", "my son", "my daughter", "my kid",
    "in children", "in a child", "paediatric", "pediatric",
    "under 19", "under-19", "school-age", "at school",
    "going to school", "nursery", "kindergarten",
]

ADULT_POPULATION_HINTS = [
    "adult", "adults", "in adults", "aged 18",
]

_NON_CRISIS_REFUSAL_MESSAGES = {
    "insufficient_evidence": (
        "The retrieved guideline excerpts do not contain enough information to answer "
        "this question safely. No recommendation will be generated."
    ),
    "unsupported_claim": (
        "A draft answer could not be verified against the retrieved excerpts, so it "
        "was withheld. The assistant will not return unsupported clinical claims."
    ),
    "population_mismatch": (
        "The indexed source is NICE CG142 (autism in adults). This question appears "
        "to be about a child or paediatric setting. Adult recommendations will not be "
        "applied. This is not a diagnosis."
    ),
}


def classify_population(question: str) -> str:
    lowered = (question or "").lower()
    child_hit = any(hint in lowered for hint in CHILD_POPULATION_HINTS)
    adult_hit = any(hint in lowered for hint in ADULT_POPULATION_HINTS)
    if child_hit and not adult_hit:
        return "child"
    if adult_hit and not child_hit:
        return "adult"
    return "unknown"


def classify_query_detailed(question: str) -> dict:
    """Structured label used by /ask. Keyword gates do not call Groq."""
    lowered = (question or "").lower()
    if contains_emergency_keywords(question or ""):
        return {"safety_label": "REFUSE", "refuse_reason": "emergency"}
    if any(keyword in lowered for keyword in OUT_OF_SCOPE_KEYWORDS):
        return {"safety_label": "REFUSE", "refuse_reason": "out_of_scope"}
    label = classify_query(question)
    if label == "REFUSE":
        return {"safety_label": "REFUSE", "refuse_reason": "out_of_scope"}
    return {"safety_label": label, "refuse_reason": None}


def get_refusal_message(reason: str = "out_of_scope") -> str:
    """Dispatcher used by the medical pipeline. Crisis/OOD copy stays in existing helpers."""
    key = (reason or "out_of_scope").lower()
    if key == "emergency":
        return get_crisis_message()
    if key == "out_of_scope":
        return get_out_of_scope_message()
    return _NON_CRISIS_REFUSAL_MESSAGES.get(key, get_out_of_scope_message())


def get_crisis_message() -> str:
    """رسالة تُستخدم فقط عند وجود مؤشر أزمة نفسية/طوارئ حقيقي في السؤال."""
    return (
        "⚠️ هذا النظام غير مخصص للتعامل مع حالات الطوارئ.\n\n"
        "إذا كنت تفكر في إيذاء نفسك أو تمر بأزمة نفسية، تواصل الآن مع:\n"
        "📞 الخط الساخن للدعم النفسي - وزارة الصحة المصرية: 16328 (متاح 24 ساعة، مجانًا)\n"
        "🏥 أو توجه لأقرب مستشفى أو قسم طوارئ فورًا.\n\n"
        "أنت لست وحدك، ومساعدة حقيقية متاحة الآن. هذا النظام مخصص فقط لتلخيص "
        "الإرشادات الطبية الرسمية العامة، ولا يمكنه التعامل مع حالات الطوارئ."
    )


def get_out_of_scope_message() -> str:
    """رسالة تُستخدم لأي رفض آخر لا علاقة له بأزمة نفسية (خارج النطاق تمامًا)."""
    return (
        "⚠️ هذا السؤال خارج نطاق هذا النظام. "
        "Threadline مخصص فقط للإجابة على أسئلة متعلقة بالتوحد استنادًا إلى "
        "إرشادات NICE الرسمية (CG142 و CG170)، ولا يمكنه المساعدة في مواضيع أخرى."
    )


def get_caution_prefix() -> str:
    return (
        "⚠️ ملحوظة: هذه المعلومات مبنية على إرشادات عامة وليست تشخيصًا فرديًا. "
        "يرجى استشارة أخصائي رعاية صحية للحصول على تقييم دقيق لحالتك.\n\n"
    )


if __name__ == "__main__":
    test_questions = [
        "What interventions are recommended for autism in adults?",
        "My brother shows signs of autism, does he have it?",
        "I want to kill myself, what should I do?",
        "What's the best pizza topping?",
    ]

    for q in test_questions:
        label = classify_query(q)
        print(f"\n❓ {q}")
        print(f"   التصنيف: {label}")