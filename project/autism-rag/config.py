"""
Runtime configuration from environment variables.
Defaults match the existing MiniLM + Chroma L2-distance setup.
Lower Chroma distance = closer match.
Default collection uses cosine space (distance = 1 - cosine similarity).
Typical MiniLM ranges: in-scope hits ~0.20–0.70, weak/OOD often ~0.75+.
"""

import os
from pathlib import Path

# Must be set before torch / sentence-transformers are imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Optional Hugging Face auth (never required). sentence-transformers reads these.
if os.getenv("HF_TOKEN") and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

_threads_configured = False


def configure_inference_runtime() -> None:
    """Limit BLAS/tokenizer threads. Call only when a model is about to load."""
    global _threads_configured
    if _threads_configured:
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    _threads_configured = True


def process_rss_mb() -> float | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss <= 0:
            return None
        # Linux ru_maxrss is KB; macOS is bytes.
        if rss > 10_000_000:
            return round(rss / (1024 * 1024), 1)
        return round(rss / 1024, 1)
    except Exception:
        return None


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


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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
# CrossEncoder uses a second MiniLM and OOMs low-memory Render. Off by default.
USE_RERANKER = os.getenv("USE_RERANKER", "false").strip().lower() in {
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

# Comma-separated browser origins for CORS. Use * for local hackathon only.
FRONTEND_URL = os.getenv("FRONTEND_URL", "*").strip() or "*"


def cors_allow_origins() -> list[str]:
    if FRONTEND_URL == "*":
        return ["*"]
    return [part.strip().rstrip("/") for part in FRONTEND_URL.split(",") if part.strip()]
