from config import CHROMA_COLLECTION, CHROMA_DB_PATH, EMBEDDING_MODEL, RETRIEVAL_DISTANCE_THRESHOLD

_model = None


def get_embedding_model(model_name: str | None = None):
    from sentence_transformers import SentenceTransformer

    global _model
    name = model_name or EMBEDDING_MODEL
    if _model is None or getattr(_model, "_rag_model_name", None) != name:
        _model = SentenceTransformer(name)
        _model._rag_model_name = name
    return _model


def load_collection(db_path: str | None = None, collection_name: str | None = None):
    import chromadb

    client = chromadb.PersistentClient(path=db_path or CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=collection_name or CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _meta_str(meta: dict, key: str, default: str = "") -> str:
    value = meta.get(key, default)
    return default if value is None else str(value)


def retrieve(query_text: str, collection, model, top_k: int = 5):
    """
    Return the closest top_k chunks with full metadata and Chroma distance.
    Lower distance = closer semantic match.
    """
    if collection.count() == 0:
        return []

    n_results = min(top_k, collection.count())
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    retrieved = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        retrieved.append({
            "text": doc or "",
            "page_number": meta.get("page_number"),
            "section_title": _meta_str(meta, "section_title", "General"),
            "chunk_id": _meta_str(meta, "chunk_id"),
            "document_name": _meta_str(meta, "document_name"),
            "source_url": _meta_str(meta, "source_url"),
            "distance": float(distance) if distance is not None else None,
        })
    return retrieved


def retrieval_quality_gate(
    retrieved: list[dict],
    threshold: float | None = None,
) -> dict:
    """
    Pass when at least one chunk exists and the best (lowest) distance
    is at or below RETRIEVAL_DISTANCE_THRESHOLD.
    """
    limit = RETRIEVAL_DISTANCE_THRESHOLD if threshold is None else threshold
    if not retrieved:
        return {
            "passed": False,
            "best_distance": None,
            "threshold": limit,
            "reason": "no_results",
        }

    distances = [r["distance"] for r in retrieved if r.get("distance") is not None]
    if not distances:
        return {
            "passed": False,
            "best_distance": None,
            "threshold": limit,
            "reason": "no_scores",
        }

    best = min(distances)
    passed = best <= limit
    return {
        "passed": passed,
        "best_distance": best,
        "threshold": limit,
        "reason": "ok" if passed else "above_threshold",
    }


def serialize_source(chunk: dict) -> dict:
    page = chunk.get("page_number")
    try:
        page = int(page) if page is not None else 0
    except (TypeError, ValueError):
        page = 0

    distance = chunk.get("distance")
    try:
        distance = float(distance) if distance is not None else 0.0
    except (TypeError, ValueError):
        distance = 0.0

    return {
        "chunk_id": chunk.get("chunk_id") or "",
        "document": chunk.get("document_name") or "",
        "source_url": chunk.get("source_url") or "",
        "section": chunk.get("section_title") or "General",
        "page": page,
        "distance": distance,
        "text": chunk.get("text") or "",
    }


def get_index_stats(collection=None) -> dict:
    collection = collection or load_collection()
    count = collection.count()
    documents = {}
    if count:
        data = collection.get(include=["metadatas"])
        for meta in data.get("metadatas") or []:
            name = (meta or {}).get("document_name") or "Unknown"
            documents[name] = documents.get(name, 0) + 1
    return {
        "chunk_count": count,
        "documents": [
            {"name": name, "chunks": n}
            for name, n in sorted(documents.items())
        ],
    }


def print_results(query_text: str, results: list[dict]):
    print(f"\n🔍 السؤال: '{query_text}'")
    for i, r in enumerate(results):
        print(
            f"\n  [{i+1}] {r.get('chunk_id')} | {r.get('document_name', '')[:40]} | "
            f"صفحة {r['page_number']} | {r['section_title']} | distance={r['distance']:.4f}"
        )
        print(f"      {r['text'][:150]}")


if __name__ == "__main__":
    collection = load_collection()
    model = get_embedding_model()

    test_query = "What medication should not be used for core features of autism?"

    for k in [3, 5]:
        results = retrieve(test_query, collection, model, top_k=k)
        print(f"\n{'='*60}\nTop-{k} results\n{'='*60}")
        print_results(test_query, results)
        print("quality gate:", retrieval_quality_gate(results))
