"""
Step 6: Grounded Generation & Citation
=========================================
الهدف: ناخد الـ chunks اللي رجعت من البحث (Step 4) ونخلي الـ LLM يرد
على السؤال بناءً على النص ده بس - من غير ما يضيف أي معلومة من "معرفته
العامة" (اللي ممكن تكون غلط أو مش من مصدر موثوق).

المبدأ الأساسي: النص المسترجع = مصدر الحقيقة الوحيد. الـ LLM دوره
إنه "يلخص ويرتب" مش إنه "يفكر ويجاوب من عنده".
"""

import os
from dotenv import load_dotenv
from groq import Groq

from step4_retrieval import load_collection, retrieve
from sentence_transformers import SentenceTransformer

load_dotenv()  # بيقرأ ملف .env ويحمّل المفتاح في متغيرات البيئة

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# الـ system prompt ده أهم جزء في الملف كله - هو اللي بيمنع الـ "هلوسة"
SYSTEM_PROMPT = """You are a clinical evidence assistant. You answer questions
ONLY using the provided guideline excerpts below. You must NEVER use any
external medical knowledge, even if you are confident it is correct.

STRICT RULES:
1. If the provided excerpts contain a clear answer, summarize it concisely
   and cite the exact page number and section for every claim you make.
2. If the excerpts do NOT contain enough information to answer the question,
   you MUST respond with: "INSUFFICIENT_EVIDENCE" followed by a brief
   explanation of what is missing. Do NOT guess or fill gaps with your own
   knowledge.
3. Never provide a diagnosis or medical advice for a specific individual.
   You only summarize what the guideline says in general.
4. Format your answer as:
   RECOMMENDATION: <concise summary>
   EVIDENCE: <short supporting excerpt(s), quoted from the provided text>
   CITATION: <document name, section, page number for each claim>
   CONFIDENCE: <High / Medium / Low / Insufficient Evidence>
"""


def build_context(retrieved_chunks: list[dict]) -> str:
    """
    بنحوّل الـ chunks المسترجعة لنص واحد منظم، كل جزء معاه مصدره بوضوح،
    عشان الموديل يقدر يستشهد بيه بدقة.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Excerpt {i}] (Page {chunk['page_number']}, Section: {chunk['section_title']})\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = build_context(retrieved_chunks)

    user_message = f"""Question: {question}

Provided guideline excerpts:
{context}

Answer the question using ONLY the excerpts above, following the strict
rules and format given in your instructions."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # موديل سريع ومجاني مناسب لضغط وقت الهاكاثون
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # قيمة منخفضة جدًا عشان الرد يكون ثابت ومحافظ، مش "مبدع"
    )
    return response.choices[0].message.content


def ask(question: str, top_k: int = 5):
    """
    الدالة الكاملة end-to-end: سؤال -> بحث -> توليد إجابة مبنية على الأدلة
    """
    collection = load_collection()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    retrieved = retrieve(question, collection, model, top_k=top_k)
    answer = generate_answer(question, retrieved)

    print(f"\n{'='*70}")
    print(f"❓ السؤال: {question}")
    print(f"{'='*70}")
    print(answer)
    print(f"\n📚 المصادر المستخدمة في البحث:")
    for r in retrieved:
        print(f"   - صفحة {r['page_number']} | {r['section_title']} | distance={r['distance']:.4f}")

    return answer


if __name__ == "__main__":
    # جرب سؤال له إجابة واضحة في المصدر
    ask("Should chelation be used for managing autism symptoms?")

    # جرب سؤال خارج نطاق المستند تمامًا - المفروض يرفض بأمانة
    ask("What is the recommended treatment protocol for autism in cats?")