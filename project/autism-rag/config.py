"""
Runtime configuration from environment variables.
Defaults match the existing MiniLM + Chroma L2-distance setup.
Lower Chroma distance = closer match.
Default collection uses cosine space (distance = 1 - cosine similarity).
Typical MiniLM ranges: in-scope hits ~0.20–0.70, weak/OOD often ~0.75+.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EVAL_ALT_EMBEDDING_MODEL = os.getenv("EVAL_ALT_EMBEDDING_MODEL", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Refuse generation when the best retrieved distance is worse (higher) than this.
# Calibrated on cosine MiniLM distances after reindex: in-scope top hits were
# ~0.16–0.50; unrelated trivia ~0.92. Semantic near-misses (e.g. "autism in cats")
# can still score ~0.48 and must be caught by classification/grounding, not this gate.
RETRIEVAL_DISTANCE_THRESHOLD = _float("RETRIEVAL_DISTANCE_THRESHOLD", 0.70)

TOP_K_DEFAULT = _int("TOP_K_DEFAULT", 5)

CHUNK_MIN_TOKENS = _int("CHUNK_MIN_TOKENS", 120)
CHUNK_MAX_TOKENS = _int("CHUNK_MAX_TOKENS", 800)
CHUNK_OVERLAP_TOKENS = _int("CHUNK_OVERLAP_TOKENS", 80)

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "autism_nice_cg142")

RETRIEVAL_LOG_PATH = os.getenv("RETRIEVAL_LOG_PATH", "retrieval_logs.jsonl")

# Lexical coverage of recommendation tokens that must appear in retrieved text.
GROUNDING_TOKEN_COVERAGE = _float("GROUNDING_TOKEN_COVERAGE", 0.45)

# Hybrid retrieval (dense Chroma + BM25 + RRF + cross-encoder).
USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
HYBRID_CANDIDATE_COUNT = _int("HYBRID_CANDIDATE_COUNT", 20)
RRF_K = _int("RRF_K", 60)
RERANK_FINAL_COUNT = _int("RERANK_FINAL_COUNT", 3)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Initial hackathon calibration for ms-marco MiniLM logits (higher = more relevant).
# Not a validated clinical cutoff. 0.0 false-abstains labeled hits with negative logits.
RERANK_SCORE_THRESHOLD = _float("RERANK_SCORE_THRESHOLD", -2.0)
RETRIEVAL_DEBUG = os.getenv("RETRIEVAL_DEBUG", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
CHUNKS_JSON_PATH = os.getenv("CHUNKS_JSON_PATH", "step2_chunks.json")
