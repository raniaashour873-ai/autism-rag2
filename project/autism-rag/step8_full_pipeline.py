"""
Step 8: Full Pipeline (End-to-End)
"""

import json
from datetime import datetime, timezone

from config import RETRIEVAL_DEBUG, RETRIEVAL_LOG_PATH, TOP_K_DEFAULT, USE_HYBRID_RETRIEVAL
from hybrid_retrieval import hybrid_retrieve_and_rerank
from step4_retrieval import (
    get_embedding_model,
    load_collection,
    retrieval_quality_gate,
    retrieve,
    serialize_source,
)
from step6_generation import (
    format_display_answer,
    generate_answer,
    grounding_check,
    resolve_citations,
)
from step7_safety import (
    classify_population,
    classify_query_detailed,
    get_caution_prefix,
    get_refusal_message,
)


def _empty_result(status, safety_label, refuse_reason, answer, sources=None, extra=None):
    payload = {
        "status": status,
        "safety_label": safety_label,
        "refuse_reason": refuse_reason,
        "answer": answer,
        "recommendation": "",
        "evidence": "",
        "citations": [],
        "sources": sources or [],
        "generation_called": False,
    }
    if extra:
        payload.update(extra)
    return payload


def _log_retrieval(record: dict) -> None:
    try:
        with open(RETRIEVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def full_pipeline(question: str, top_k: int | None = None) -> dict:
    top_k = TOP_K_DEFAULT if top_k is None else top_k
    question = (question or "").strip()

    if not question:
        return _empty_result(
            "refused",
            "REFUSE",
            "out_of_scope",
            "A clinical question is required.",
        )

    classification = classify_query_detailed(question)
    safety_label = classification["safety_label"]
    refuse_reason = classification["refuse_reason"]

    if safety_label == "REFUSE":
        result = _empty_result(
            "refused",
            safety_label,
            refuse_reason,
            get_refusal_message(refuse_reason or "out_of_scope"),
        )
        _log_retrieval({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": question,
            "top_k": top_k,
            "retrieved_chunk_ids": [],
            "distances": [],
            "refuse_reason": refuse_reason,
            "safety_label": safety_label,
            "status": "refused",
            "generation_called": False,
            "confidence_passed": None,
        })
        return result

    population = classify_population(question)
    if population == "child":
        result = _empty_result(
            "refused",
            "REFUSE",
            "population_mismatch",
            get_refusal_message("population_mismatch"),
        )
        _log_retrieval({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": question,
            "top_k": top_k,
            "retrieved_chunk_ids": [],
            "refuse_reason": "population_mismatch",
            "safety_label": "REFUSE",
            "population": population,
            "status": "refused",
            "generation_called": False,
        })
        return result

    collection = load_collection()
    model = get_embedding_model()

    hybrid_debug = {}
    if USE_HYBRID_RETRIEVAL:
        packed = hybrid_retrieve_and_rerank(
            question,
            collection,
            model,
            final_k=top_k,
        )
        retrieved = packed["selected"]
        gate = {
            "passed": not packed["abstain"],
            "reason": packed["reason"],
            "threshold": packed["threshold"],
            "best_rerank_score": packed["all_scored"][0]["rerank_score"] if packed["all_scored"] else None,
        }
        hybrid_debug = {
            "dense_count": packed.get("dense_count"),
            "bm25_count": packed.get("bm25_count"),
            "fused_count": len(packed.get("fused") or []),
            "reranked_ids": [c.get("chunk_id") for c in packed.get("all_scored") or []],
            "rerank_scores": [
                {"chunk_id": c.get("chunk_id"), "rerank_score": c.get("rerank_score")}
                for c in (packed.get("all_scored") or [])[:10]
            ],
            "final_chunk_ids": [c.get("chunk_id") for c in retrieved],
            "abstain": packed["abstain"],
        }
        if RETRIEVAL_DEBUG:
            print("[retrieval-debug]", json.dumps({
                "query": question,
                **hybrid_debug,
            }, ensure_ascii=False))
    else:
        retrieved = retrieve(question, collection, model, top_k=top_k)
        gate = retrieval_quality_gate(retrieved)

    sources = [serialize_source(r) for r in retrieved]

    log_base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": question,
        "top_k": top_k,
        "retrieved_chunk_ids": [r.get("chunk_id") for r in retrieved],
        "distances": [r.get("distance") for r in retrieved],
        "best_distance": gate.get("best_distance"),
        "best_rerank_score": gate.get("best_rerank_score"),
        "threshold": gate.get("threshold"),
        "confidence_passed": gate.get("passed"),
        "safety_label": safety_label,
        "hybrid": hybrid_debug or None,
    }

    if not gate["passed"]:
        result = _empty_result(
            "refused",
            "REFUSE",
            "insufficient_evidence",
            get_refusal_message("insufficient_evidence"),
            sources=sources,
            extra={"generation_called": False, "retrieval_gate": gate},
        )
        _log_retrieval({**log_base, "status": "refused", "refuse_reason": "insufficient_evidence", "generation_called": False})
        return result

    generated = generate_answer(question, retrieved)

    if generated["status"] != "answered":
        result = _empty_result(
            "refused",
            "REFUSE",
            "insufficient_evidence",
            get_refusal_message("insufficient_evidence"),
            sources=sources,
            extra={"generation_called": True, "retrieval_gate": gate},
        )
        _log_retrieval({**log_base, "status": "refused", "refuse_reason": "insufficient_evidence", "generation_called": True})
        return result

    cited = resolve_citations(generated["citation_ids"], retrieved)
    if generated["citation_ids"] and not cited:
        result = _empty_result(
            "refused",
            "REFUSE",
            "insufficient_evidence",
            get_refusal_message("insufficient_evidence"),
            sources=sources,
            extra={"generation_called": True, "retrieval_gate": gate},
        )
        _log_retrieval({**log_base, "status": "refused", "refuse_reason": "invalid_citation_ids", "generation_called": True})
        return result

    if not cited:
        result = _empty_result(
            "refused",
            "REFUSE",
            "insufficient_evidence",
            get_refusal_message("insufficient_evidence"),
            sources=sources,
            extra={"generation_called": True, "retrieval_gate": gate},
        )
        _log_retrieval({**log_base, "status": "refused", "refuse_reason": "missing_citation_ids", "generation_called": True})
        return result

    ground = grounding_check(
        generated["recommendation"],
        generated["evidence"],
        retrieved,
    )
    if not ground["passed"]:
        result = _empty_result(
            "refused",
            "REFUSE",
            "unsupported_claim",
            get_refusal_message("unsupported_claim"),
            sources=sources,
            extra={"generation_called": True, "retrieval_gate": gate, "grounding": ground},
        )
        _log_retrieval({**log_base, "status": "refused", "refuse_reason": "unsupported_claim", "generation_called": True, "grounding": ground})
        return result

    answer = format_display_answer(
        generated["recommendation"],
        generated["evidence"],
        cited,
    )
    if safety_label == "NEEDS_CAUTION":
        answer = get_caution_prefix() + answer

    citations = [serialize_source(c) for c in cited]
    result = {
        "status": "answered",
        "safety_label": safety_label,
        "refuse_reason": None,
        "answer": answer,
        "recommendation": generated["recommendation"],
        "evidence": generated["evidence"],
        "citations": citations,
        "sources": sources,
        "generation_called": True,
        "retrieval_gate": gate,
        "grounding": ground,
    }
    _log_retrieval({**log_base, "status": "answered", "refuse_reason": None, "generation_called": True, "citation_ids": generated["citation_ids"]})
    return result


if __name__ == "__main__":
    test_questions = [
        "Should chelation be used for managing autism symptoms?",
        "My son shows repetitive behaviors, does he have autism?",
        "I want to kill myself, what should I do?",
        "What is the recommended treatment protocol for autism in cats?",
    ]

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        print(f"{'='*70}")
        result = full_pipeline(q)
        print(f"Status: {result['status']} | Safety: {result['safety_label']} | Reason: {result.get('refuse_reason')}")
        print(f"generation_called: {result.get('generation_called')}")
        print(f"\n{result['answer']}")
        if result["sources"]:
            print("\n📚 المصادر:")
            for s in result["sources"]:
                print(f"   - {s['chunk_id']} | {s['document'][:40]} | p.{s['page']} | {s['section']}")
