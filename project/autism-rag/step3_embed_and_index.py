import json

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_COLLECTION, CHROMA_DB_PATH, EMBEDDING_MODEL

DOCUMENT_NAME = "NICE CG142 - Autism spectrum disorder in adults: diagnosis and management"
SOURCE_URL = "https://www.nice.org.uk/guidance/cg142"


def build_metadata(chunk: dict) -> dict:
    return {
        "document_name": DOCUMENT_NAME,
        "page_number": chunk["page_number"],
        "section_title": chunk["section_title"],
        "chunk_id": chunk["chunk_id"],
        "source_url": SOURCE_URL,
    }


def run_step3(
    chunks_path: str = "step2_chunks.json",
    db_path: str | None = None,
    collection_name: str | None = None,
    reset: bool = True,
):
    db_path = db_path or CHROMA_DB_PATH
    collection_name = collection_name or CHROMA_COLLECTION

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"عدد الـ chunks المطلوب فهرستها: {len(chunks)}")

    print("بنحمّل موديل الـ embeddings (أول مرة بياخد وقت أطول)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=db_path)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"تم حذف المجموعة القديمة: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    metadatas = [build_metadata(c) for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    print("Generating embeddings with EMBEDDING_MODEL (must match query model)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"\n✅ تم فهرسة {collection.count()} chunk في: {db_path}")
    return collection


def test_query(collection, query_text: str, top_k: int = 3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    print(f"\n🔍 نتائج البحث عن: '{query_text}'")
    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        print(f"\n  النتيجة {i+1} (المسافة: {distance:.4f}) — {meta.get('chunk_id')} | صفحة {meta['page_number']} | {meta['section_title']}")
        print(f"  {doc[:150]}")


if __name__ == "__main__":
    collection = run_step3(reset=True)
    test_query(collection, "Do not use biological tests for diagnosis")
