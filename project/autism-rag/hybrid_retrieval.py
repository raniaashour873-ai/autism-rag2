"""
BM25 + dense hybrid retrieval with RRF and optional cross-encoder rerank.

Dense-only retrieve() in step4_retrieval.py is unchanged and used as one
leg of hybrid fusion.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config import (
    CHUNKS_JSON_PATH,
    HYBRID_CANDIDATE_COUNT,
    RERANK_FINAL_COUNT,
    RERANK_SCORE_THRESHOLD,
    RERANKER_MODEL,
    RETRIEVAL_DEBUG,
    RRF_K,
)
from step4_retrieval import retrieve

logger = logging.getLogger("autism_rag.retrieval")
if RETRIEVAL_DEBUG:
    logging.basicConfig(level=logging.DEBUG)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*", re.IGNORECASE)

_bm25 = None
_bm25_chunks: list[dict] = []
_bm25_ids: list[str] = []
_reranker = None

DOCUMENT_NAME = "NICE CG142 - Autism spectrum disorder in adults: diagnosis and management"
SOURCE_URL = "https://www.nice.org.uk/guidance/cg142"


def _chunks_path() -> Path:
    path = Path(CHUNKS_JSON_PATH)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _normalize_chunk(raw: dict) -> dict:
    return {
        "chunk_id": str(raw.get("chunk_id") or ""),
        "text": raw.get("text") or "",
        "page_number": raw.get("page_number"),
        "section_title": raw.get("section_title") or "General",
        "document_name": raw.get("document_name") or DOCUMENT_NAME,
        "source_url": raw.get("source_url") or SOURCE_URL,
        "distance": raw.get("distance"),
        "bm25_score": raw.get("bm25_score"),
        "rrf_score": raw.get("rrf_score"),
        "rerank_score": raw.get("rerank_score"),
        "dense_rank": raw.get("dense_rank"),
        "bm25_rank": raw.get("bm25_rank"),
    }


def load_bm25_corpus(chunks_path: str | Path | None = None) -> list[dict]:
    """Load step2_chunks.json once for BM25. Does not touch Chroma."""
    global _bm25, _bm25_chunks, _bm25_ids
    if _bm25 is not None:
        return _bm25_chunks

    from rank_bm25 import BM25Okapi

    path = Path(chunks_path) if chunks_path else _chunks_path()
    with open(path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    _bm25_chunks = [_normalize_chunk(c) for c in raw_chunks]
    _bm25_ids = [c["chunk_id"] for c in _bm25_chunks]
    tokenized = [_tokenize(c["text"]) for c in _bm25_chunks]
    _bm25 = BM25Okapi(tokenized)
    logger.debug("BM25 index built over %s chunks from %s", len(_bm25_chunks), path)
    return _bm25_chunks


def bm25_retrieve(query_text: str, top_k: int = 20, chunks_path: str | Path | None = None) -> list[dict]:
    load_bm25_corpus(chunks_path)
    if _bm25 is None or not _bm25_chunks:
        return []

    scores = _bm25.get_scores(_tokenize(query_text))
    ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), _bm25_ids[item[0]]))
    n = min(top_k, len(ranked))
    results = []
    for rank, (idx, score) in enumerate(ranked[:n], start=1):
        chunk = dict(_bm25_chunks[idx])
        chunk["bm25_score"] = float(score)
        chunk["bm25_rank"] = rank
        results.append(chunk)
    return results


def rrf_fuse(rankings: list[list[str]], rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    """
    rankings: lists of chunk_id in rank order (best first).
    Returns (chunk_id, rrf_score) sorted by score desc, then chunk_id for stability.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def hybrid_retrieve(
    query_text: str,
    collection,
    model,
    candidate_count: int | None = None,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Dense retrieve() + BM25, fused with RRF. Does not rerank."""
    n = candidate_count or HYBRID_CANDIDATE_COUNT
    dense = retrieve(query_text, collection, model, top_k=n)
    sparse = bm25_retrieve(query_text, top_k=n)

    by_id: dict[str, dict] = {}
    dense_ids = []
    for i, chunk in enumerate(dense, start=1):
        cid = chunk.get("chunk_id") or ""
        if not cid:
            continue
        dense_ids.append(cid)
        merged = _normalize_chunk(chunk)
        merged["dense_rank"] = i
        by_id[cid] = merged

    bm25_ids = []
    for i, chunk in enumerate(sparse, start=1):
        cid = chunk.get("chunk_id") or ""
        if not cid:
            continue
        bm25_ids.append(cid)
        if cid in by_id:
            by_id[cid]["bm25_score"] = chunk.get("bm25_score")
            by_id[cid]["bm25_rank"] = i
        else:
            extra = _normalize_chunk(chunk)
            extra["bm25_rank"] = i
            by_id[cid] = extra

    fused = rrf_fuse([dense_ids, bm25_ids], rrf_k=rrf_k)
    results = []
    for cid, score in fused[:n]:
        chunk = dict(by_id[cid])
        chunk["rrf_score"] = score
        results.append(chunk)

    if RETRIEVAL_DEBUG:
        logger.debug(
            "hybrid query=%r dense=%s bm25=%s fused=%s ids=%s",
            query_text,
            len(dense_ids),
            len(bm25_ids),
            len(results),
            [c["chunk_id"] for c in results],
        )
    return results


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.debug("Loaded reranker %s", RERANKER_MODEL)
    return _reranker


def rerank_chunks(
    query_text: str,
    candidates: list[dict],
    final_k: int | None = None,
    score_threshold: float | None = None,
    reranker=None,
) -> dict:
    """
    Score query+text with a cross-encoder. Returns:
      all_scored, passed (threshold), selected (top final_k of passed).
    """
    keep = RERANK_FINAL_COUNT if final_k is None else final_k
    limit = RERANK_SCORE_THRESHOLD if score_threshold is None else score_threshold

    if not candidates:
        return {"all_scored": [], "passed": [], "selected": [], "abstain": True, "reason": "no_candidates"}

    model = reranker if reranker is not None else get_reranker()
    pairs = [[query_text, c.get("text") or ""] for c in candidates]
    scores = model.predict(pairs)
    scored = []
    for chunk, score in zip(candidates, scores):
        item = dict(chunk)
        item["rerank_score"] = float(score)
        scored.append(item)
    scored.sort(key=lambda c: (-c["rerank_score"], c.get("chunk_id") or ""))

    passed = [c for c in scored if c["rerank_score"] >= limit]
    selected = passed[:keep]
    abstain = len(selected) == 0
    if RETRIEVAL_DEBUG:
        logger.debug(
            "rerank query=%r candidates=%s passed=%s selected=%s abstain=%s scores=%s",
            query_text,
            len(scored),
            len(passed),
            [c["chunk_id"] for c in selected],
            abstain,
            [(c["chunk_id"], round(c["rerank_score"], 4)) for c in scored[:8]],
        )
    return {
        "all_scored": scored,
        "passed": passed,
        "selected": selected,
        "abstain": abstain,
        "reason": "below_threshold" if abstain else "ok",
        "threshold": limit,
    }


def hybrid_retrieve_and_rerank(
    query_text: str,
    collection,
    model,
    final_k: int | None = None,
    candidate_count: int | None = None,
    score_threshold: float | None = None,
    reranker=None,
) -> dict:
    fused = hybrid_retrieve(
        query_text,
        collection,
        model,
        candidate_count=candidate_count,
    )
    reranked = rerank_chunks(
        query_text,
        fused,
        final_k=final_k,
        score_threshold=score_threshold,
        reranker=reranker,
    )
    reranked["fused"] = fused
    reranked["dense_count"] = sum(1 for c in fused if c.get("dense_rank"))
    reranked["bm25_count"] = sum(1 for c in fused if c.get("bm25_rank"))
    return reranked
