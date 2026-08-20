import logging
import threading

from config import (
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    RETRIEVAL_DISTANCE_THRESHOLD,
    configure_inference_runtime,
    process_rss_mb,
)

logger = logging.getLogger("autism_rag.retrieval")

_model = None
_model_lock = threading.Lock()
_chroma_client = None
_collection = None
_collection_lock = threading.Lock()
_stats_cache = None


_MINILM_NAMES = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
}


def get_embedding_model(model_name: str | None = None):
    """Production query encoder is ONNX MiniLM. PyTorch is local-only and never a fallback."""
    global _model
    name = model_name or EMBEDDING_MODEL
    backend = (EMBEDDING_BACKEND or "onnx").strip().lower()
    if _model is not None and getattr(_model, "_rag_model_name", None) == name:
        return _model
    with _model_lock:
        if _model is not None and getattr(_model, "_rag_model_name", None) == name:
            return _model
        configure_inference_runtime()
        if backend == "onnx":
            if name not in _MINILM_NAMES:
                raise RuntimeError(
                    f"ONNX encoder only supports all-MiniLM-L6-v2, got {name!r}. "
                    "Set EMBEDDING_BACKEND=pytorch for local indexing/eval only."
                )
            from onnx_encoder import get_onnx_encoder

            _model = get_onnx_encoder()
            _model._rag_model_name = name
            return _model
        if backend == "pytorch":
            from sentence_transformers import SentenceTransformer

            logger.info("[MEM] before embedding initialization model=%s (rss_mb=%s)", name, process_rss_mb())
            _model = SentenceTransformer(name, device="cpu")
            _model._rag_model_name = name
            logger.info("[MEM] after embedding initialization (rss_mb=%s)", process_rss_mb())
            return _model
        raise RuntimeError(
            f"Unknown EMBEDDING_BACKEND={backend!r}. Use onnx (production) or pytorch (local only)."
        )


def load_collection(db_path: str | None = None, collection_name: str | None = None):
    import chromadb

    global _chroma_client, _collection
    path = db_path or CHROMA_DB_PATH
    name = collection_name or CHROMA_COLLECTION
    if (
        _collection is not None
        and db_path is None
        and collection_name is None
    ):
        return _collection
    with _collection_lock:
        if (
            _collection is not None
            and db_path is None
            and collection_name is None
        ):
            return _collection
        logger.info("Opening Chroma PersistentClient path=%s (rss_mb=%s)", path, process_rss_mb())
        _chroma_client = chromadb.PersistentClient(path=path)
        # Never create/rebuild the index from the query path.
        _collection = _chroma_client.get_collection(name=name)
        return _collection


def _meta_str(meta: dict, key: str, default: str = "") -> str:
    value = meta.get(key, default)
    return default if value is None else str(value)


def retrieve(query_text: str, collection, model, top_k: int = 5):
    """
    Return the closest top_k chunks with full metadata and Chroma distance.
    Lower distance = closer semantic match.
    """
    total = collection.count()
    if total == 0:
        return []

    n_results = min(top_k, total)
    logger.info("[MEM] before query encoding (rss_mb=%s)", process_rss_mb())
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    logger.info("[MEM] after query encoding dim=%s (rss_mb=%s)", len(query_embedding[0]), process_rss_mb())
    logger.info("[MEM] before Chroma query n_results=%s (rss_mb=%s)", n_results, process_rss_mb())
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    logger.info("[MEM] after Chroma query (rss_mb=%s)", process_rss_mb())

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
    """
    Optional stats helper. Not used on /health. Avoids embedding model load.
    Caches one metadata scan so callers do not reload the full collection.
    """
    global _stats_cache
    if collection is None and _stats_cache is not None:
        return _stats_cache
    collection = collection or load_collection()
    count = collection.count()
    documents = {}
    if count:
        data = collection.get(include=["metadatas"])
        for meta in data.get("metadatas") or []:
            name = (meta or {}).get("document_name") or "Unknown"
            documents[name] = documents.get(name, 0) + 1
    stats = {
        "chunk_count": count,
        "documents": [
            {"name": name, "chunks": n}
            for name, n in sorted(documents.items())
        ],
    }
    if collection is _collection:
        _stats_cache = stats
    return stats


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
