"""
Step 6: Grounded Generation & Citation
"""

import json
import os
import re

from dotenv import load_dotenv

from config import EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL, GROUNDING_TOKEN_COVERAGE

load_dotenv()

_client = None


def _groq_client():
    from groq import Groq

    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY or os.environ.get("GROQ_API_KEY"))
    return _client

SYSTEM_PROMPT = """You are a clinical evidence assistant. You answer questions
ONLY using the provided guideline excerpts. You must NEVER use any
external medical knowledge, even if you are confident it is correct.

STRICT RULES:
1. Use ONLY the excerpt text. Do not add facts that are not written there.
2. If the excerpts do not contain enough information to answer, set
   status to "insufficient_evidence". Do not guess.
3. Never diagnose or advise a specific individual. Only summarize the guideline.
4. Cite evidence using chunk_id values from the excerpts (e.g. "chunk_0042").
   NEVER invent page numbers, document names, or section titles.
5. evidence must be a short quote copied from the provided excerpts.
6. recommendation may only restate what the excerpts say.

Return ONLY valid JSON with this schema:
{
  "status": "answered" | "insufficient_evidence",
  "recommendation": "string",
  "evidence": "string",
  "citation_ids": ["chunk_xxxx"]
}
"""

_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "should", "would",
    "could", "about", "into", "their", "there", "have", "been", "were",
    "will", "also", "only", "than", "then", "them", "they", "what", "when",
    "which", "while", "using", "used", "does", "not", "are", "was", "but",
}


def build_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[chunk_id={chunk['chunk_id']}]\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_generation_output(raw: str) -> dict:
    """Parse model output into a structured dict. Fail closed to insufficient_evidence."""
    insufficient = {
        "status": "insufficient_evidence",
        "recommendation": "",
        "evidence": "",
        "citation_ids": [],
        "raw": raw,
    }
    if not raw:
        return insufficient

    stripped = _strip_json_fence(raw)

    parsed = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        if "INSUFFICIENT_EVIDENCE" in stripped.upper():
            return insufficient
        return insufficient

    status = str(parsed.get("status", "")).strip().lower()
    if status in {"insufficient_evidence", "insufficient", "refuse"}:
        status = "insufficient_evidence"
    elif status != "answered":
        status = "insufficient_evidence"

    citation_ids = parsed.get("citation_ids") or parsed.get("citations") or []
    if isinstance(citation_ids, str):
        citation_ids = re.findall(r"chunk_[0-9A-Za-z_]+", citation_ids)
    elif not isinstance(citation_ids, list):
        citation_ids = []
    citation_ids = [str(cid).strip() for cid in citation_ids if str(cid).strip()]

    return {
        "status": status,
        "recommendation": str(parsed.get("recommendation") or "").strip(),
        "evidence": str(parsed.get("evidence") or "").strip(),
        "citation_ids": citation_ids,
        "raw": raw,
    }


def resolve_citations(citation_ids: list[str], retrieved_chunks: list[dict]) -> list[dict]:
    by_id = {c["chunk_id"]: c for c in retrieved_chunks if c.get("chunk_id")}
    resolved = []
    seen = set()
    for cid in citation_ids:
        if cid in seen:
            continue
        chunk = by_id.get(cid)
        if not chunk:
            continue
        seen.add(cid)
        resolved.append(chunk)
    return resolved


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def grounding_check(
    recommendation: str,
    evidence_quote: str,
    retrieved_chunks: list[dict],
    min_coverage: float | None = None,
) -> dict:
    """
    Lightweight lexical check: content words in the recommendation (and quote)
    should appear in retrieved excerpt text. This is not an NLI model.
    """
    limit = GROUNDING_TOKEN_COVERAGE if min_coverage is None else min_coverage
    context = "\n".join(c.get("text") or "" for c in retrieved_chunks)
    context_tokens = _content_tokens(context)
    claim_tokens = _content_tokens(recommendation) | _content_tokens(evidence_quote)

    if not claim_tokens:
        return {"passed": False, "coverage": 0.0, "unsupported_tokens": [], "reason": "empty_claim"}

    if not context_tokens:
        return {"passed": False, "coverage": 0.0, "unsupported_tokens": sorted(claim_tokens), "reason": "empty_context"}

    missing = sorted(t for t in claim_tokens if t not in context_tokens)
    coverage = 1.0 - (len(missing) / len(claim_tokens))
    passed = coverage >= limit
    return {
        "passed": passed,
        "coverage": coverage,
        "unsupported_tokens": missing[:20],
        "reason": "ok" if passed else "low_coverage",
    }


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    context = build_context(retrieved_chunks)
    ids = ", ".join(c["chunk_id"] for c in retrieved_chunks)

    user_message = f"""Question: {question}

Allowed chunk_id values: {ids}

Provided guideline excerpts:
{context}

Return JSON only."""

    response = _groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    return parse_generation_output(raw)


def format_display_answer(recommendation: str, evidence: str, citations: list[dict]) -> str:
    citation_lines = []
    for c in citations:
        citation_lines.append(
            f"{c.get('document_name') or c.get('document')}, "
            f"{c.get('section_title') or c.get('section')}, "
            f"p.{c.get('page_number') or c.get('page')} "
            f"({c.get('chunk_id')})"
        )
    citation_block = "\n".join(citation_lines) if citation_lines else "None (unresolved)"
    return (
        f"RECOMMENDATION: {recommendation}\n\n"
        f"EVIDENCE: {evidence}\n\n"
        f"CITATION:\n{citation_block}"
    )


def ask(question: str, top_k: int = 5):
    from step4_retrieval import get_embedding_model, load_collection, retrieve

    collection = load_collection()
    model = get_embedding_model()
    retrieved = retrieve(question, collection, model, top_k=top_k)
    parsed = generate_answer(question, retrieved)
    print(parsed)
    return parsed


if __name__ == "__main__":
    ask("Should chelation be used for managing autism symptoms?")
    ask("What is the recommended treatment protocol for autism in cats?")
