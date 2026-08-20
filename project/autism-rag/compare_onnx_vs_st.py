"""Local-only check: ONNX MiniLM vs SentenceTransformer on the existing Chroma index.

Does not write to chroma_db. Requires requirements-dev.txt (PyTorch) for the ST side.
"""

from __future__ import annotations

import math

import numpy as np
from sentence_transformers import SentenceTransformer

from onnx_encoder import MiniLMOnnxEncoder
from step4_retrieval import load_collection, retrieve

QUERY = "What medication should not be used for core features of autism?"
ST_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    st = SentenceTransformer(ST_NAME, device="cpu")
    onnx = MiniLMOnnxEncoder()

    st_vec = st.encode([QUERY], convert_to_numpy=True, show_progress_bar=False)[0]
    onnx_vec = onnx.encode([QUERY], convert_to_numpy=True)[0]

    print("ST dim:", len(st_vec), "finite:", bool(np.isfinite(st_vec).all()))
    print("ONNX dim:", len(onnx_vec), "finite:", bool(np.isfinite(onnx_vec).all()))
    print("ST L2:", float(np.linalg.norm(st_vec)))
    print("ONNX L2:", float(np.linalg.norm(onnx_vec)))
    sim = cosine(st_vec, onnx_vec)
    print("cosine(ST, ONNX):", sim)
    if len(st_vec) != 384 or len(onnx_vec) != 384:
        raise SystemExit("dimension mismatch")
    if not math.isfinite(sim) or sim < 0.99:
        raise SystemExit(f"cosine too low: {sim}")

    collection = load_collection()
    if collection.count() != 63:
        print("WARN collection count:", collection.count(), "expected 63")

    st_hits = retrieve(QUERY, collection, st, top_k=5)
    onnx_hits = retrieve(QUERY, collection, onnx, top_k=5)
    print("ST top-5:", [(h["chunk_id"], round(h["distance"], 6)) for h in st_hits])
    print("ONNX top-5:", [(h["chunk_id"], round(h["distance"], 6)) for h in onnx_hits])
    st_ids = [h["chunk_id"] for h in st_hits]
    onnx_ids = [h["chunk_id"] for h in onnx_hits]
    overlap = len(set(st_ids) & set(onnx_ids))
    print("top-5 id overlap:", overlap, "/ 5")
    if st_ids != onnx_ids:
        print("WARN ranking differs; overlap still", overlap)
    if overlap < 4:
        raise SystemExit("top-5 ranking too far from ST")
    print("PASS")


if __name__ == "__main__":
    main()
