"""
Step 7: Input Risk Classification (Safety Layer)
"""

import os

from dotenv import load_dotenv

from config import GROQ_API_KEY, GROQ_MODEL

load_dotenv()
_client = None


def _groq_client():
    from groq import Groq

    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY or os.environ.get("GROQ_API_KEY"))
    return _client

CLASSIFICATION_PROMPT = """You are a safety classifier for a clinical guideline
assistant. Classify the following user question into exactly ONE category:

- ALLOWED: A general question about autism diagnosis, management, or care
  practices as covered in clinical guidelines.

- NEEDS_CAUTION: A question about a specific individual's personal situation
  or symptoms. These can still be answered with general guideline information,
  with a disclaimer that this is not an individual diagnosis.

- REFUSE_EMERGENCY: Medical emergency, immediate danger, suicide, self-harm,
  overdose, or a request for a specific drug dosage for an individual.

- REFUSE_OUT_OF_SCOPE: Completely unrelated to autism clinical guidelines
  (other species, unrelated topics, general trivia, etc.).

Respond with ONLY one token:
ALLOWED, NEEDS_CAUTION, REFUSE_EMERGENCY, or REFUSE_OUT_OF_SCOPE.
"""

EMERGENCY_KEYWORDS = [
    "suicide", "kill myself", "self-harm", "harm myself",
    "overdose", "emergency", "dying", "chest pain", "can't breathe",
    "kill himself", "kill herself", "want to die",
]

OUT_OF_SCOPE_KEYWORDS = [
    "pizza", "best topping", "stock market", "football score",
    " in cats", " in dogs", " in hamsters", "veterinary",
    "for cats", "for dogs",
]

REFUSAL_MESSAGES = {
    "emergency": (
        "This system is not for medical emergencies or individual crisis care. "
        "If you or someone else is in immediate danger, contact local emergency "
        "services or go to the nearest hospital now. This assistant only summarizes "
        "official clinical guidelines for general information."
    ),
    "out_of_scope": (
        "This question is outside the scope of the indexed autism clinical guidelines. "
        "The assistant will not answer from general knowledge."
    ),
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

# Caregiver/child wording — retrieval corpus is adult CG142 only.
CHILD_POPULATION_HINTS = [
    "my child", "my son", "my daughter", "my kid",
    "in children", "in a child", "paediatric", "pediatric",
    "under 19", "under-19", "school-age", "at school",
    "going to school", "nursery", "kindergarten",
]

ADULT_POPULATION_HINTS = [
    "adult", "adults", "in adults", "aged 18",
]


def classify_population(question: str) -> str:
    """
    Lightweight adult/child/unknown tag for corpus mismatch.
    Does not diagnose autism. Generic guideline questions stay 'unknown'.
    """
    lowered = (question or "").lower()
    child_hit = any(hint in lowered for hint in CHILD_POPULATION_HINTS)
    adult_hit = any(hint in lowered for hint in ADULT_POPULATION_HINTS)
    if child_hit and not adult_hit:
        return "child"
    if adult_hit and not child_hit:
        return "adult"
    return "unknown"


def classify_query(question: str) -> str:
    """Backward-compatible label: ALLOWED / NEEDS_CAUTION / REFUSE."""
    return classify_query_detailed(question)["safety_label"]


def classify_query_detailed(question: str) -> dict:
    lowered = (question or "").lower()
    if any(keyword in lowered for keyword in EMERGENCY_KEYWORDS):
        return {
            "safety_label": "REFUSE",
            "refuse_reason": "emergency",
        }
    if any(keyword in lowered for keyword in OUT_OF_SCOPE_KEYWORDS):
        return {
            "safety_label": "REFUSE",
            "refuse_reason": "out_of_scope",
        }

    response = _groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    label = (response.choices[0].message.content or "").strip().upper()
    label = label.replace(" ", "_").split()[0] if label else ""

    if label in {"REFUSE_EMERGENCY", "EMERGENCY"}:
        return {"safety_label": "REFUSE", "refuse_reason": "emergency"}
    if label in {"REFUSE_OUT_OF_SCOPE", "OUT_OF_SCOPE", "REFUSE"}:
        return {"safety_label": "REFUSE", "refuse_reason": "out_of_scope"}
    if label == "ALLOWED":
        return {"safety_label": "ALLOWED", "refuse_reason": None}
    if label == "NEEDS_CAUTION":
        return {"safety_label": "NEEDS_CAUTION", "refuse_reason": None}

    return {"safety_label": "NEEDS_CAUTION", "refuse_reason": None}


def get_refusal_message(reason: str = "out_of_scope") -> str:
    return REFUSAL_MESSAGES.get(reason, REFUSAL_MESSAGES["out_of_scope"])


def get_caution_prefix() -> str:
    return (
        "Note: This information is from general clinical guidelines and is not "
        "an individual diagnosis. Consult a qualified healthcare professional "
        "for personal assessment.\n\n"
    )


if __name__ == "__main__":
    test_questions = [
        "What interventions are recommended for autism in adults?",
        "My brother shows signs of autism, does he have it?",
        "I want to kill myself, what should I do?",
        "What's the best pizza topping?",
    ]

    for q in test_questions:
        result = classify_query_detailed(q)
        print(f"\n❓ {q}")
        print(f"   {result}")
