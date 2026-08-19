"""
Situation → NICE Evidence → Structured Actionable Routine
"""
import json
import os
from dotenv import load_dotenv
from groq import Groq

from step4_retrieval import load_collection, retrieve
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ROUTINE_PROMPT = """You are a clinical evidence assistant that turns NICE
guideline excerpts into a short, practical routine for a caregiver.

STRICT RULES:
1. Use ONLY the provided excerpts as the basis for the routine. Do not add
   any external knowledge or invent recommendations not grounded in the text.
2. If the excerpts do not contain enough relevant guidance to build a
   routine, respond with exactly: {"insufficient_evidence": true}
3. Output ONLY valid JSON, no markdown, no extra text, in this exact shape:
{
  "goal": "short goal title",
  "steps": [
    {"icon": "single relevant emoji", "title": "short step title", "detail": "one sentence, grounded in the excerpts"}
  ],
  "citations": ["Section X.X, Page N", ...]
}
Use 3 to 6 steps maximum.
"""

def generate_routine(situation: str, top_k: int = 5) -> dict:
    collection = load_collection()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    retrieved = retrieve(situation, collection, model, top_k=top_k)

    context = "\n---\n".join(
        f"[Excerpt {i+1}] (Page {r['page_number']}, Section: {r['section_title']})\n{r['text']}"
        for i, r in enumerate(retrieved)
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": ROUTINE_PROMPT},
            {"role": "user", "content": f"Caregiver situation: {situation}\n\nGuideline excerpts:\n{context}"},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"insufficient_evidence": True}

    data["sources"] = [
        {"page": r["page_number"], "section": r["section_title"], "distance": r["distance"]}
        for r in retrieved
    ]
    return data


if __name__ == "__main__":
    result = generate_routine("My son shows repetitive behaviors and struggles with changes to his daily schedule")
    print(json.dumps(result, indent=2, ensure_ascii=False))