import json
import os
import re

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EVAL_ALT_EMBEDDING_MODEL, GROQ_API_KEY, HYBRID_CANDIDATE_COUNT
from hybrid_retrieval import hybrid_retrieve_and_rerank
from step4_retrieval import get_embedding_model, load_collection, retrieve
from step6_generation import grounding_check
from step8_full_pipeline import full_pipeline


def _is_hit(item: dict, results: list[dict]) -> tuple[bool, int | None, str | None]:
    retrieved_ids = [r["chunk_id"] for r in results]
    expected_id = item.get("expected_chunk_id")
    needle = (item.get("expected_contains") or "").strip().lower()
    expected_page = item.get("expected_page")

    if expected_id and expected_id in retrieved_ids:
        return True, retrieved_ids.index(expected_id) + 1, expected_id

    if needle:
        for i, result in enumerate(results):
            if needle in (result.get("text") or "").lower():
                return True, i + 1, result["chunk_id"]

    if expected_page is not None:
        for i, result in enumerate(results):
            if result.get("page_number") == expected_page and needle:
                if needle[:40] in (result.get("text") or "").lower():
                    return True, i + 1, result["chunk_id"]

    rec_id = (item.get("expected_recommendation_id") or "").strip()
    if rec_id:
        pattern = re.compile(rf"(?<![\d.]){re.escape(rec_id)}(?![\d])", re.IGNORECASE)
        for i, result in enumerate(results):
            blob = f"{result.get('section_title') or ''} {result.get('text') or ''}"
            if pattern.search(blob):
                return True, i + 1, result["chunk_id"]

    return False, None, retrieved_ids[0] if retrieved_ids else None


def evaluate_precision_at_k(eval_path: str = "eval_questions.json", top_k: int = 5):
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    collection = load_collection()
    model = SentenceTransformer(EMBEDDING_MODEL)

    hits = 0
    total_scored = 0
    detailed_results = []

    for item in eval_set:
        qtype = item.get("type") or "direct"
        results = retrieve(item["question"], collection, model, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]
        top_distance = results[0]["distance"] if results else None

        if qtype == "direct":
            total_scored += 1
            is_hit, rank, matched_id = _is_hit(item, results)
            if is_hit:
                hits += 1
            detailed_results.append({
                "question": item["question"],
                "type": qtype,
                "expected": item.get("expected_chunk_id"),
                "expected_contains": item.get("expected_contains"),
                "retrieved_ids": retrieved_ids,
                "retrieved_top1": retrieved_ids[0] if retrieved_ids else None,
                "top_distance": top_distance,
                "hit": is_hit,
                "rank": rank,
                "matched_id": matched_id,
            })
        else:
            detailed_results.append({
                "question": item["question"],
                "type": qtype,
                "retrieved_ids": retrieved_ids,
                "top_distance": top_distance,
                "note": item.get("note") or (
                    "Should be refused or answered only from retrieved evidence."
                    if qtype == "ambiguous"
                    else "Out-of-scope queries should not produce a grounded clinical recommendation."
                ),
            })

    precision_at_k = hits / total_scored if total_scored else 0

    print(f"\n{'='*60}")
    print(f"Precision@{top_k} = {hits}/{total_scored} = {precision_at_k:.2%}")
    print(f"{'='*60}\n")

    for row in detailed_results:
        if row["type"] != "direct":
            print(f"⚠️  [{row['type'].upper()}] {row['question']}")
            dist = row["top_distance"]
            print(f"    top distance: {dist:.4f}" if dist is not None else "    no results")
            print(f"    {row['note']}\n")
            continue
        status = "✅" if row["hit"] else "❌"
        print(f"{status} {row['question']}")
        print(
            f"    expected contains: {row['expected_contains']!r} | "
            f"top-1: {row['retrieved_top1']} | rank: {row['rank']} | "
            f"distance: {row['top_distance']}\n"
        )

    return {
        "precision_at_k": precision_at_k,
        "k": top_k,
        "hits": hits,
        "total": total_scored,
        "details": detailed_results,
    }


def _metrics_from_ranks(ranks: list[int | None], k: int) -> dict:
    scored = len(ranks)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    mrr = 0.0
    if scored:
        mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / scored
    return {
        "precision_at_k": hits / scored if scored else 0,
        "recall_at_k": hits / scored if scored else 0,
        "mrr": mrr,
        "hits": hits,
        "total": scored,
        "k": k,
    }


def evaluate_retrieval_comparison(eval_path: str = "eval_questions.json", k: int = 5):
    """Dense-only retrieve() vs hybrid RRF + cross-encoder rerank. No Groq."""
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    collection = load_collection()
    model = get_embedding_model()
    rows = []

    for item in eval_set:
        if (item.get("type") or "direct") != "direct":
            continue
        question = item["question"]
        current = retrieve(question, collection, model, top_k=k)
        improved_pack = hybrid_retrieve_and_rerank(
            question,
            collection,
            model,
            final_k=k,
            candidate_count=max(HYBRID_CANDIDATE_COUNT, k),
            score_threshold=float("-inf"),
        )
        improved = improved_pack["all_scored"][:k]

        cur_hit, cur_rank, _ = _is_hit(item, current)
        imp_hit, imp_rank, _ = _is_hit(item, improved)
        rows.append({
            "question": question,
            "expected_recommendation_id": item.get("expected_recommendation_id"),
            "current_rank": cur_rank,
            "improved_rank": imp_rank,
            "current_hit": cur_hit,
            "improved_hit": imp_hit,
            "current_top1": current[0]["chunk_id"] if current else None,
            "improved_top1": improved[0]["chunk_id"] if improved else None,
        })

    current_ranks = [r["current_rank"] for r in rows]
    improved_ranks = [r["improved_rank"] for r in rows]
    current_metrics = _metrics_from_ranks(current_ranks, k)
    improved_metrics = _metrics_from_ranks(improved_ranks, k)

    print(f"\n{'='*80}")
    print(f"CURRENT (dense-only) vs IMPROVED (BM25+dense+RRF+rerank)  k={k}")
    print(f"{'='*80}")
    print(
        f"{'Query':<72} {'Cur':>5} {'Imp':>5} {'C':>3} {'I':>3}"
    )
    for row in rows:
        q = row["question"][:70]
        print(
            f"{q:<72} {str(row['current_rank'] or '-'):>5} "
            f"{str(row['improved_rank'] or '-'):>5} "
            f"{'Y' if row['current_hit'] else 'N':>3} "
            f"{'Y' if row['improved_hit'] else 'N':>3}"
        )
        if "biological or genetic" in row["question"].lower():
            print(
                f"  ** known miss tracker: current_hit={row['current_hit']} "
                f"improved_hit={row['improved_hit']} "
                f"current_top1={row['current_top1']} improved_top1={row['improved_top1']}"
            )

    print(
        f"\nCURRENT  P@{k}={current_metrics['precision_at_k']:.2%}  "
        f"R@{k}={current_metrics['recall_at_k']:.2%}  MRR={current_metrics['mrr']:.3f}"
    )
    print(
        f"IMPROVED P@{k}={improved_metrics['precision_at_k']:.2%}  "
        f"R@{k}={improved_metrics['recall_at_k']:.2%}  MRR={improved_metrics['mrr']:.3f}"
    )
    return {
        "k": k,
        "current": current_metrics,
        "improved": improved_metrics,
        "rows": rows,
        "note": (
            "Recall@k equals Precision@k here because each labeled question has one "
            "relevant recommendation identity. Rerank comparison uses score_threshold=-inf "
            "so ranking is measured even if the serving abstention threshold would drop a chunk."
        ),
    }


def evaluate_citation_accuracy(eval_path: str = "eval_questions.json"):
    """
    Deterministic: cited chunk_id must exist in the retrieved set, and
    displayed document/section/page must match that chunk's metadata.
    """
    if not GROQ_API_KEY and not os.environ.get("GROQ_API_KEY"):
        print("Skipping citation accuracy eval (no GROQ_API_KEY).")
        return {"skipped": True, "reason": "GROQ_API_KEY missing"}

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    scored = 0
    correct = 0
    details = []

    for item in eval_set:
        if item.get("type") != "direct":
            continue
        result = full_pipeline(item["question"], top_k=5)
        retrieved_by_id = {s["chunk_id"]: s for s in result.get("sources") or []}
        citations = result.get("citations") or []
        item_ok = True
        reasons = []

        if result["status"] != "answered":
            item_ok = False
            reasons.append(f"status={result['status']} reason={result.get('refuse_reason')}")
        elif not citations:
            item_ok = False
            reasons.append("no citations")
        else:
            for cite in citations:
                source = retrieved_by_id.get(cite.get("chunk_id"))
                if not source:
                    item_ok = False
                    reasons.append(f"citation {cite.get('chunk_id')} not in retrieved sources")
                    continue
                if cite.get("page") != source.get("page"):
                    item_ok = False
                    reasons.append("page mismatch")
                if cite.get("section") != source.get("section"):
                    item_ok = False
                    reasons.append("section mismatch")
                if cite.get("document") != source.get("document"):
                    item_ok = False
                    reasons.append("document mismatch")

        scored += 1
        if item_ok:
            correct += 1
        details.append({
            "question": item["question"],
            "ok": item_ok,
            "reasons": reasons,
            "citation_ids": [c.get("chunk_id") for c in citations],
        })
        mark = "✅" if item_ok else "❌"
        print(f"{mark} citation | {item['question']}")
        if reasons:
            print(f"    {reasons}")

    accuracy = correct / scored if scored else 0
    print(f"\nCitation accuracy = {correct}/{scored} = {accuracy:.2%}")
    return {
        "citation_accuracy": accuracy,
        "correct": correct,
        "total": scored,
        "details": details,
        "methodology": (
            "For each in-scope question, run the full pipeline. Every returned "
            "citation must reference a retrieved chunk_id, and document/section/page "
            "must equal that chunk's metadata. The LLM is not asked to score itself."
        ),
    }


def evaluate_faithfulness(eval_path: str = "eval_questions.json"):
    """
    Lexical coverage of recommendation/evidence tokens against retrieved excerpts.
    Same check used at request time (not an NLI model).
    """
    if not GROQ_API_KEY and not os.environ.get("GROQ_API_KEY"):
        print("Skipping faithfulness eval (no GROQ_API_KEY).")
        return {"skipped": True, "reason": "GROQ_API_KEY missing"}

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    scored = 0
    faithful = 0
    details = []

    for item in eval_set:
        if item.get("type") != "direct":
            continue
        result = full_pipeline(item["question"], top_k=5)
        scored += 1
        if result["status"] != "answered":
            # Refusal is treated as faithful (no unsupported claim was returned).
            faithful += 1
            details.append({
                "question": item["question"],
                "faithful": True,
                "status": result["status"],
                "refuse_reason": result.get("refuse_reason"),
                "note": "refusal counted as no unsupported claim returned",
            })
            continue

        retrieved = [
            {
                "text": s.get("text") or "",
                "chunk_id": s.get("chunk_id"),
            }
            for s in result.get("sources") or []
        ]
        check = grounding_check(
            result.get("recommendation") or "",
            result.get("evidence") or "",
            retrieved,
        )
        if check["passed"]:
            faithful += 1
        details.append({
            "question": item["question"],
            "faithful": check["passed"],
            "coverage": check["coverage"],
            "unsupported_tokens": check.get("unsupported_tokens"),
            "status": result["status"],
        })
        mark = "✅" if check["passed"] else "❌"
        print(f"{mark} faithfulness | coverage={check['coverage']:.2f} | {item['question']}")

    rate = faithful / scored if scored else 0
    print(f"\nFaithfulness (lexical coverage) = {faithful}/{scored} = {rate:.2%}")
    return {
        "faithfulness": rate,
        "faithful": faithful,
        "total": scored,
        "details": details,
        "methodology": (
            "Content words in recommendation+evidence quote must appear in retrieved "
            "chunk text. Coverage threshold is GROUNDING_TOKEN_COVERAGE. Refusals are "
            "counted as faithful because no unsupported recommendation was shown. "
            "This is a deterministic overlap check, not a neural NLI score."
        ),
    }


def evaluate_embedding_models(eval_path: str = "eval_questions.json", top_k: int = 5):
    alt = EVAL_ALT_EMBEDDING_MODEL
    if not alt:
        print(
            "Embedding model comparison skipped. Set EVAL_ALT_EMBEDDING_MODEL "
            "to a sentence-transformers model name to compare against "
            f"{EMBEDDING_MODEL} without changing the default serving model."
        )
        return {
            "skipped": True,
            "default_model": EMBEDDING_MODEL,
            "note": "Optional: set EVAL_ALT_EMBEDDING_MODEL then re-run this function.",
        }

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    directs = [i for i in eval_set if i.get("type") == "direct"]
    collection = load_collection()

    report = {"default_model": EMBEDDING_MODEL, "alt_model": alt, "k": top_k, "models": {}}
    for name in [EMBEDDING_MODEL, alt]:
        model = SentenceTransformer(name)
        hits = 0
        for item in directs:
            results = retrieve(item["question"], collection, model, top_k=top_k)
            ok, _, _ = _is_hit(item, results)
            if ok:
                hits += 1
        precision = hits / len(directs) if directs else 0
        report["models"][name] = {"hits": hits, "total": len(directs), "precision_at_k": precision}
        print(f"{name}: Precision@{top_k} = {hits}/{len(directs)} = {precision:.2%}")
        print("NOTE: alt-model query embeddings are compared against the existing MiniLM index,")
        print("so this is only a fair comparison if the collection was built with the same model.")
    return report


def save_results(payload: dict, path: str = "evaluation_results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run citation-accuracy and faithfulness (requires GROQ_API_KEY).",
    )
    args = parser.parse_args()

    p3 = evaluate_precision_at_k(top_k=3)
    p5 = evaluate_precision_at_k(top_k=5)
    comparison = evaluate_retrieval_comparison(k=5)
    citation = {"skipped": True, "reason": "pass --full to run"}
    faithfulness = {"skipped": True, "reason": "pass --full to run"}
    if args.full:
        citation = evaluate_citation_accuracy()
        faithfulness = evaluate_faithfulness()
    embedding = evaluate_embedding_models()
    save_results({
        "precision_at_3": p3,
        "precision_at_5": p5,
        "retrieval_comparison": comparison,
        "citation_accuracy": citation,
        "faithfulness": faithfulness,
        "embedding_comparison": embedding,
    })
