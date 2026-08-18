import json
from step4_retrieval import load_collection, retrieve
from sentence_transformers import SentenceTransformer


def evaluate_precision_at_k(eval_path: str = "eval_questions.json", top_k: int = 5):
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    collection = load_collection()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    hits = 0
    total_scored = 0  # هنعد بس الأسئلة اللي ليها إجابة متوقعة (نستبعد out_of_scope من حساب الدقة)
    detailed_results = []

    for item in eval_set:
        results = retrieve(item["question"], collection, model, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]

        if item["type"] == "direct":
            total_scored += 1
            is_hit = item["expected_chunk_id"] in retrieved_ids
            if is_hit:
                hits += 1
            detailed_results.append({
                "question": item["question"],
                "expected": item["expected_chunk_id"],
                "retrieved_top1": retrieved_ids[0] if retrieved_ids else None,
                "hit": is_hit,
                "rank": retrieved_ids.index(item["expected_chunk_id"]) + 1 if is_hit else None
            })
        else:
            # out-of-scope: بنسجل بس أقرب مسافة رجعت، عشان نشوف هل النظام "متردد" فعلاً
            top_distance = results[0]["distance"] if results else None
            detailed_results.append({
                "question": item["question"],
                "type": "out_of_scope",
                "top_distance": top_distance,
                "note": "لازم النظام يرفض الإجابة هنا حتى لو رجع chunks، لأن مفيش إجابة حقيقية"
            })

    precision_at_k = hits / total_scored if total_scored else 0

    print(f"\n{'='*60}")
    print(f"Precision@{top_k} = {hits}/{total_scored} = {precision_at_k:.2%}")
    print(f"{'='*60}\n")

    for r in detailed_results:
        if r.get("type") == "out_of_scope":
            print(f"⚠️  [OUT-OF-SCOPE] {r['question']}")
            print(f"    أقرب distance رجع: {r['top_distance']:.4f} (لو الرقم ده كبير، معناه صح إن مفيش تطابق حقيقي)\n")
        else:
            status = "✅" if r["hit"] else "❌"
            print(f"{status} {r['question']}")
            print(f"    متوقع: {r['expected']} | رجع فعليًا (top-1): {r['retrieved_top1']} | rank: {r['rank']}\n")

    # نحفظ النتائج التفصيلية عشان نستخدمها في الـ dashboard بعدين
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "precision_at_k": precision_at_k,
            "k": top_k,
            "hits": hits,
            "total": total_scored,
            "details": detailed_results
        }, f, ensure_ascii=False, indent=2)

    return precision_at_k


if __name__ == "__main__":
    print("--- Precision@3 ---")
    evaluate_precision_at_k(top_k=3)

    print("\n--- Precision@5 ---")
    evaluate_precision_at_k(top_k=5)