"""
Routine adaptation helper for autistic users.

Isolated from the medical RAG pipeline: no Chroma, BM25, reranking, or
NICE excerpt retrieval. Uses the existing Groq client only.
"""

from __future__ import annotations

import json

from config import GROQ_MODEL
from step6_generation import _groq_client, _strip_json_fence
from step7_safety import EMERGENCY_KEYWORDS, get_crisis_message

ROUTINE_DISCLAIMER = (
    "This is general routine-planning support, not a diagnosis, treatment plan, "
    "or individualized clinical, occupational therapy, or caregiver prescription. "
    "It does not replace a qualified professional."
)

ROUTINE_PROMPT = """You help an autistic adult adapt an existing daily routine.

This is NOT medical care. Do not diagnose autism. Do not recommend medication,
tests, treatments, or clinical interventions. Do not claim the routine is
prescribed by NICE or any guideline. Do not invent the user's clinical needs.

If the user described a routine, adapt THAT routine. Prefer small, gradual
changes over replacing everything at once.

Helpful patterns (use only when they fit what the user said):
- break activities into smaller steps
- clearer sequencing
- predictable transitions and brief warnings before a change
- checklist-style wording
- rough time estimates and buffer time
- optional breaks
- sensory considerations ONLY if the user mentioned them
- "if this step becomes difficult" alternatives

Return ONLY valid JSON:
{
  "goal": "short title for this adapted routine",
  "steps": [
    {"title": "short step name", "detail": "one or two practical sentences", "time_estimate": "optional e.g. 5–10 min"}
  ],
  "transitions": ["before changing from X to Y, ..."],
  "if_difficult": ["option if a step is hard"]
}
Use 4 to 10 steps. Keep language plain and actionable.
"""


def _empty_routine(*, status: str, refuse_reason: str | None, message: str, generation_called: bool) -> dict:
    return {
        "status": status,
        "refuse_reason": refuse_reason,
        "message": message,
        "disclaimer": ROUTINE_DISCLAIMER,
        "goal": "",
        "steps": [],
        "transitions": [],
        "if_difficult": [],
        "generation_called": generation_called,
        "used_medical_retrieval": False,
    }


def generate_routine(
    situation: str = "",
    current_routine: str = "",
    difficulties: str = "",
    preferred_structure: str = "",
    time_available: str = "",
    things_that_help: str = "",
) -> dict:
    parts = {
        "What they want help with": (situation or "").strip(),
        "Current routine": (current_routine or "").strip(),
        "What feels difficult": (difficulties or "").strip(),
        "Preferred structure": (preferred_structure or "").strip(),
        "Time available": (time_available or "").strip(),
        "Things that help": (things_that_help or "").strip(),
    }
    combined = " ".join(parts.values()).lower()
    if not combined.strip():
        return _empty_routine(
            status="refused",
            refuse_reason="out_of_scope",
            message="Describe a routine you want help adapting.",
            generation_called=False,
        )

    if any(keyword in combined for keyword in EMERGENCY_KEYWORDS):
        return _empty_routine(
            status="refused",
            refuse_reason="emergency",
            message=get_crisis_message(),
            generation_called=False,
        )

    user_block = "\n".join(f"{label}: {value}" for label, value in parts.items() if value)

    response = _groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": ROUTINE_PROMPT},
            {"role": "user", "content": user_block},
        ],
        temperature=0.3,
    )
    raw = _strip_json_fence(response.choices[0].message.content or "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _empty_routine(
            status="refused",
            refuse_reason="insufficient_evidence",
            message="The routine helper could not produce a usable plan. Try describing the current steps more simply.",
            generation_called=True,
        )

    steps = data.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return _empty_routine(
            status="refused",
            refuse_reason="insufficient_evidence",
            message="The routine helper could not produce steps. Try listing the current order of activities.",
            generation_called=True,
        )

    cleaned_steps = []
    for step in steps[:10]:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "").strip()
        if not title:
            continue
        cleaned_steps.append({
            "title": title,
            "detail": str(step.get("detail") or "").strip(),
            "time_estimate": str(step.get("time_estimate") or "").strip(),
        })

    transitions = [str(x).strip() for x in (data.get("transitions") or []) if str(x).strip()]
    if_difficult = [str(x).strip() for x in (data.get("if_difficult") or []) if str(x).strip()]

    return {
        "status": "adapted",
        "refuse_reason": None,
        "message": "",
        "disclaimer": ROUTINE_DISCLAIMER,
        "goal": str(data.get("goal") or "Adapted routine").strip(),
        "steps": cleaned_steps,
        "transitions": transitions,
        "if_difficult": if_difficult,
        "generation_called": True,
        "used_medical_retrieval": False,
    }


if __name__ == "__main__":
    print(json.dumps(generate_routine(
        situation="I have a morning routine: wake up, breakfast, shower, get dressed, leave home. I struggle with sudden transitions and getting ready on time.",
    ), indent=2, ensure_ascii=False))
