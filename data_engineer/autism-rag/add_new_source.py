"""
Add New Source: يضيف أي مستند PDF جديد لنفس قاعدة البيانات الموجودة
=======================================================================
بيستخدم نفس منطق step1 و step2 و step3، بس بيتقبل اسم مستند ورابط
مختلفين عشان نقدر نفرق بين مصادر متعددة (بالغين/أطفال) في نفس الـ DB.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

from step1_extract_pdf import extract_pdf_pages, clean_text, find_boilerplate_lines
from step2_chunking import split_into_sections, chunk_section, estimate_tokens


def process_new_pdf(pdf_path: str, document_name: str, source_url: str,
                     db_path: str = "./chroma_db",
                     collection_name: str = "autism_nice_cg142",
                     min_tokens: int = 400, max_tokens: int = 800):

    print(f"[1/4] بنستخرج النص من: {pdf_path}")
    pages = extract_pdf_pages(pdf_path)
    boilerplate = find_boilerplate_lines(pages)
    cleaned_pages = []
    for p in pages:
        cleaned = clean_text(p["raw_text"], boilerplate)
        if cleaned:
            cleaned_pages.append({"page_number": p["page_number"], "text": cleaned})
    print(f"    عدد الصفحات بعد التنظيف: {len(cleaned_pages)}")

    print("[2/4] بنقسم لـ chunks...")
    all_chunks = []
    chunk_counter = 0
    for page in cleaned_pages:
        sections = split_into_sections(page["text"])
        for section in sections:
            sub_chunks = chunk_section(section["text"], min_tokens, max_tokens)
            for text_chunk in sub_chunks:
                chunk_counter += 1
                # بادئة مختلفة في الـ chunk_id عشان منتصدمش مع chunk_ids المصدر الأول
                all_chunks.append({
                    "chunk_id": f"{document_name[:6].replace(' ', '')}_{chunk_counter:04d}",
                    "page_number": page["page_number"],
                    "section_title": section["section_title"],
                    "text": text_chunk,
                })
    print(f"    عدد الـ chunks الجديدة: {len(all_chunks)}")

    print("[3/4] بنولّد الـ embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("[4/4] بنضيف لقاعدة البيانات الموجودة...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    metadatas = [
        {
            "document_name": document_name,
            "page_number": c["page_number"],
            "section_title": c["section_title"],
            "chunk_id": c["chunk_id"],
            "source_url": source_url,
        }
        for c in all_chunks
    ]
    ids = [c["chunk_id"] for c in all_chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"\n✅ تم إضافة {len(all_chunks)} chunk من '{document_name}' لنفس قاعدة البيانات")
    print(f"العدد الكلي في القاعدة الآن: {collection.count()}")


if __name__ == "__main__":
    process_new_pdf(
        pdf_path="nice_cg170.pdf",
        document_name="NICE CG170 - Autism spectrum disorder in under 19s: support and management",
        source_url="https://www.nice.org.uk/guidance/cg170",
    )