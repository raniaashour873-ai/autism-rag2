"""
Step 8: Full Pipeline (End-to-End)
=====================================
دي نقطة الدخول النهائية اللي هتستخدمها طبقة الـ Frontend بعدين.
كل سؤال بيمر على 3 مراحل بالترتيب:
  1. Safety classification -> لو REFUSE، نوقف فورًا
  2. Retrieval -> نجيب أقرب chunks
  3. Generation -> نولد إجابة مبنية على الأدلة فقط
"""

from step7_safety import classify_query, get_refusal_message, get_caution_prefix
from step4_retrieval import load_collection, retrieve
from step6_generation import generate_answer
from sentence_transformers import SentenceTransformer


def full_pipeline(question: str, top_k: int = 5) -> dict:
    # المرحلة 1: تصنيف الأمان
    safety_label = classify_query(question)

    if safety_label == "REFUSE":
        return {
            "status": "refused",
            "safety_label": safety_label,
            "answer": get_refusal_message(),
            "sources": [],
        }

    # المرحلة 2: البحث
    collection = load_collection()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    retrieved = retrieve(question, collection, model, top_k=top_k)

    # المرحلة 3: التوليد
    answer = generate_answer(question, retrieved)

    # لو الحالة تحتاج حذر، نضيف تحذير في أول الإجابة
    if safety_label == "NEEDS_CAUTION":
        answer = get_caution_prefix() + answer

    return {
        "status": "answered",
        "safety_label": safety_label,
        "answer": answer,
        "sources": [
            {"page": r["page_number"], "section": r["section_title"], "distance": r["distance"]}
            for r in retrieved
        ],
    }


if __name__ == "__main__":
    test_questions = [
        "Should chelation be used for managing autism symptoms?",
        "My son shows repetitive behaviors, does he have autism?",
        "I want to kill myself, what should I do?",
    ]

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        print(f"{'='*70}")
        result = full_pipeline(q)
        print(f"Status: {result['status']} | Safety: {result['safety_label']}")
        print(f"\n{result['answer']}")
        if result["sources"]:
            print(f"\n📚 المصادر:")
            for s in result["sources"]:
                print(f"   - صفحة {s['page']} | {s['section']}")