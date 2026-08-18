import chromadb
from sentence_transformers import SentenceTransformer


def load_collection(db_path: str = "./chroma_db", collection_name: str = "autism_nice_cg142"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)


def retrieve(query_text: str, collection, model, top_k: int = 5):
    """
    بيرجع أقرب top_k نتائج للسؤال، كل نتيجة معاها النص والـ metadata والـ score
    """
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "page_number": meta["page_number"],
            "section_title": meta["section_title"],
            "chunk_id": meta["chunk_id"],
            "distance": distance,  # كل ما الرقم أصغر، كل ما التطابق أدق
        })
    return retrieved


def print_results(query_text: str, results: list[dict]):
    print(f"\n🔍 السؤال: '{query_text}'")
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] صفحة {r['page_number']} | {r['section_title']} | distance={r['distance']:.4f}")
        print(f"      {r['text'][:150]}")


if __name__ == "__main__":
    collection = load_collection()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # جرب top-k مختلفة على نفس السؤال عشان تشوف الفرق
    test_query = "What medication should not be used for core features of autism?"

    for k in [3, 5]:
        results = retrieve(test_query, collection, model, top_k=k)
        print(f"\n{'='*60}\nTop-{k} results\n{'='*60}")
        print_results(test_query, results)