"""Offline checks that do not require Groq, except emergency keyword classification."""

from step4_retrieval import retrieval_quality_gate, serialize_source
from step6_generation import grounding_check, parse_generation_output, resolve_citations
from step7_safety import classify_query_detailed, get_refusal_message


def test_serialize_source_keeps_metadata():
    source = serialize_source({
        "chunk_id": "chunk_0007",
        "document_name": "NICE CG142 - Autism spectrum disorder in adults: diagnosis and management",
        "source_url": "https://www.nice.org.uk/guidance/cg142",
        "section_title": "1.4.14 Do not use chelation",
        "page_number": 27,
        "distance": 0.41,
        "text": "Do not use chelation for the management of core features of autism in adults.",
    })
    assert source["document"].startswith("NICE CG142")
    assert source["section"].startswith("1.4.14")
    assert source["page"] == 27
    assert "chelation" in source["text"]
    assert source["chunk_id"] == "chunk_0007"


def test_quality_gate():
    empty = retrieval_quality_gate([], threshold=0.70)
    assert empty["passed"] is False
    weak = retrieval_quality_gate([{"distance": 0.91}], threshold=0.70)
    assert weak["passed"] is False
    strong = retrieval_quality_gate([{"distance": 0.44}, {"distance": 0.60}], threshold=0.70)
    assert strong["passed"] is True
    assert weak["reason"] == "above_threshold"


def test_out_of_scope_keyword_no_llm_needed():
    result = classify_query_detailed("What is the recommended treatment protocol for autism in cats?")
    assert result["safety_label"] == "REFUSE"
    assert result["refuse_reason"] == "out_of_scope"
    pizza = classify_query_detailed("What is the best pizza topping?")
    assert pizza["refuse_reason"] == "out_of_scope"


def test_empty_question_refuses_without_generation():
    from step8_full_pipeline import full_pipeline
    result = full_pipeline("   ")
    assert result["status"] == "refused"
    assert result["generation_called"] is False


def test_ood_and_emergency_skip_generation():
    from step8_full_pipeline import full_pipeline

    ood = full_pipeline("What is the best pizza topping?")
    assert ood["status"] == "refused"
    assert ood["refuse_reason"] == "out_of_scope"
    assert ood["generation_called"] is False

    emergency = full_pipeline("I want to kill myself, what should I do?")
    assert emergency["status"] == "refused"
    assert emergency["refuse_reason"] == "emergency"
    assert emergency["generation_called"] is False
    assert "emergency" in (emergency["answer"] or "").lower()


def test_index_retrieval_gate_and_metadata():
    from step4_retrieval import get_embedding_model, load_collection, retrieve

    collection = load_collection()
    if collection.count() == 0:
        print("SKIP test_index_retrieval_gate_and_metadata (empty index)")
        return
    model = get_embedding_model()
    good = retrieve("Is chelation recommended for managing autism symptoms?", collection, model, top_k=5)
    assert good
    assert good[0].get("document_name")
    assert good[0].get("text")
    assert good[0].get("chunk_id")
    assert retrieval_quality_gate(good)["passed"] is True

    weak = retrieve("What is the best pizza topping?", collection, model, top_k=5)
    assert retrieval_quality_gate(weak)["passed"] is False



def test_parse_insufficient_and_json():
    parsed = parse_generation_output("INSUFFICIENT_EVIDENCE missing cats")
    assert parsed["status"] == "insufficient_evidence"
    parsed = parse_generation_output("""
    {
      "status": "answered",
      "recommendation": "Do not use chelation.",
      "evidence": "Do not use chelation for the management of core features",
      "citation_ids": ["chunk_0071"]
    }
    """)
    assert parsed["status"] == "answered"
    assert parsed["citation_ids"] == ["chunk_0071"]


def test_resolve_citations_drops_invalid_ids():
    retrieved = [{"chunk_id": "chunk_0071", "page_number": 27, "section_title": "A", "text": "x"}]
    resolved = resolve_citations(["chunk_9999", "chunk_0071"], retrieved)
    assert [c["chunk_id"] for c in resolved] == ["chunk_0071"]
    assert resolve_citations(["nope"], retrieved) == []


def test_grounding_detects_unsupported_tokens():
    chunks = [{"text": "Do not use chelation for the management of core features of autism in adults."}]
    ok = grounding_check("Do not use chelation for core features of autism.", "Do not use chelation", chunks)
    assert ok["passed"] is True
    bad = grounding_check(
        "Prescribe ivermectin 12 mg daily for autism as first-line therapy.",
        "ivermectin 12 mg",
        chunks,
        min_coverage=0.45,
    )
    assert bad["passed"] is False


def test_emergency_keyword_no_llm_needed():
    result = classify_query_detailed("I want to kill myself, what should I do?")
    assert result["safety_label"] == "REFUSE"
    assert result["refuse_reason"] == "emergency"
    assert "emergency" in get_refusal_message("emergency").lower()
    assert "guideline" in get_refusal_message("insufficient_evidence").lower()
    assert "outside the scope" in get_refusal_message("out_of_scope").lower()


def test_population_child_abstains_without_groq():
    from step7_safety import classify_population
    from step8_full_pipeline import full_pipeline

    assert classify_population("Should chelation be used in autistic adults?") == "adult"
    assert classify_population("What interventions are recommended for autism?") == "unknown"
    assert classify_population("My child gets upset when we change activities.") == "child"
    result = full_pipeline("My child screams every morning when it is time to go to school.")
    assert result["status"] == "refused"
    assert result["refuse_reason"] == "population_mismatch"
    assert result["generation_called"] is False


def test_bm25_and_dense_and_hybrid():
    from hybrid_retrieval import bm25_retrieve, hybrid_retrieve, rrf_fuse
    from step4_retrieval import get_embedding_model, load_collection, retrieve

    bm25 = bm25_retrieve("Do not use chelation", top_k=5)
    assert bm25
    assert bm25[0]["chunk_id"]
    assert "chelation" in (bm25[0]["text"] or "").lower() or "chelation" in (bm25[0]["section_title"] or "").lower()

    collection = load_collection()
    model = get_embedding_model()
    dense = retrieve("Is chelation recommended for managing autism symptoms?", collection, model, top_k=5)
    assert dense
    assert dense[0]["chunk_id"]

    fused = hybrid_retrieve(
        "Should biological or genetic tests be used for autism diagnosis?",
        collection,
        model,
        candidate_count=20,
    )
    ids = [c["chunk_id"] for c in fused]
    assert len(ids) == len(set(ids))
    blob = " ".join((c.get("text") or "") + " " + (c.get("section_title") or "") for c in fused)
    assert "biological tests" in blob.lower() or "1.2.11" in blob

    a = rrf_fuse([["c1", "c2"], ["c2", "c1"]], rrf_k=60)
    b = rrf_fuse([["c1", "c2"], ["c2", "c1"]], rrf_k=60)
    assert a == b
    scores = dict(a)
    assert scores["c1"] == scores["c2"]


def test_rerank_reduces_and_abstains():
    from hybrid_retrieval import rerank_chunks

    class FakeReranker:
        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]

    candidates = [{"chunk_id": f"chunk_{i:04d}", "text": f"text {i}"} for i in range(20)]
    out = rerank_chunks("q", candidates, final_k=3, score_threshold=-100, reranker=FakeReranker())
    assert len(out["selected"]) == 3
    assert [c["chunk_id"] for c in out["selected"]] == ["chunk_0019", "chunk_0018", "chunk_0017"]
    assert out["selected"][0]["chunk_id"] == candidates[-1]["chunk_id"]

    weak = rerank_chunks("q", candidates[:4], final_k=3, score_threshold=99, reranker=FakeReranker())
    assert weak["abstain"] is True
    assert weak["selected"] == []


def test_chunk_ids_survive_hybrid_pipeline():
    from hybrid_retrieval import hybrid_retrieve
    from step4_retrieval import get_embedding_model, load_collection

    collection = load_collection()
    model = get_embedding_model()
    fused = hybrid_retrieve("health passport", collection, model, candidate_count=10)
    assert fused
    for chunk in fused:
        assert chunk["chunk_id"].startswith("chunk_")
        assert chunk.get("document_name")
        assert chunk.get("section_title") is not None


def test_health_and_ask_schema():
    from fastapi.testclient import TestClient
    from api import app, AnswerResponse
    from config import RERANK_SCORE_THRESHOLD

    assert RERANK_SCORE_THRESHOLD == -2.0
    client = TestClient(app)
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["status"] == "ok"
    health = client.get("/health")
    assert health.status_code == 200
    ui = client.get("/ui")
    assert ui.status_code == 200
    html = ui.text
    assert "NICE CG142" in html
    assert "does not diagnose autism" in html.lower() or "This system does not diagnose autism" in html
    assert "does not use the medical guideline index" in html
    empty = client.post("/ask", json={"question": "   "})
    assert empty.status_code == 400
    pizza = client.post("/ask", json={"question": "What is the best pizza topping?"})
    assert pizza.status_code == 200
    body = pizza.json()
    AnswerResponse.model_validate(body)
    assert body["status"] == "refused"
    assert "sources" in body
    assert "citations" in body
    assert body["generation_called"] is False


def test_routine_endpoint_isolated_from_retrieval():
    import inspect

    import routine_generator
    from fastapi.testclient import TestClient

    from api import app

    source = inspect.getsource(routine_generator)
    assert "load_collection" not in source
    assert "hybrid_retrieve" not in source
    assert "from step4_retrieval" not in source
    assert "from hybrid_retrieval" not in source

    client = TestClient(app)
    empty = client.post("/routine", json={})
    assert empty.status_code == 400
    emergency = client.post("/routine", json={"situation": "I want to kill myself, help me plan my morning"})
    assert emergency.status_code == 200
    body = emergency.json()
    assert body["status"] == "refused"
    assert body["refuse_reason"] == "emergency"
    assert body["used_medical_retrieval"] is False
    assert body["generation_called"] is False


if __name__ == "__main__":
    tests = [
        test_serialize_source_keeps_metadata,
        test_quality_gate,
        test_parse_insufficient_and_json,
        test_resolve_citations_drops_invalid_ids,
        test_grounding_detects_unsupported_tokens,
        test_emergency_keyword_no_llm_needed,
        test_out_of_scope_keyword_no_llm_needed,
        test_empty_question_refuses_without_generation,
        test_ood_and_emergency_skip_generation,
        test_index_retrieval_gate_and_metadata,
        test_population_child_abstains_without_groq,
        test_bm25_and_dense_and_hybrid,
        test_rerank_reduces_and_abstains,
        test_chunk_ids_survive_hybrid_pipeline,
        test_health_and_ask_schema,
        test_routine_endpoint_isolated_from_retrieval,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} checks passed.")
