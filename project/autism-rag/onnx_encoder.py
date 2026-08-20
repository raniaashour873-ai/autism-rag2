"""
ONNX Runtime query encoder for sentence-transformers/all-MiniLM-L6-v2.

Matches SentenceTransformer encode() for this model:
mean pooling over tokens, then L2 normalize (the ST graph includes a Normalize
module; stored Chroma vectors have unit norm). Output dim = 384.

Does not import PyTorch or sentence-transformers.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from pathlib import Path

from config import (
    ONNX_CACHE_DIR,
    ONNX_FILENAME,
    ONNX_LOCAL_PATH,
    ONNX_MAX_LENGTH,
    ONNX_REPO,
    ONNX_TOKENIZER,
    process_rss_mb,
)

logger = logging.getLogger("autism_rag.retrieval")

_encoder = None
_encoder_lock = threading.Lock()


class OnnxEncoderError(RuntimeError):
    pass


class MiniLMOnnxEncoder:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        logger.info("[MEM] ONNX encoder loading repo=%s file=%s (rss_mb=%s)", ONNX_REPO, ONNX_FILENAME, process_rss_mb())
        try:
            cache_dir = ONNX_CACHE_DIR or None
            self.tokenizer = AutoTokenizer.from_pretrained(
                ONNX_TOKENIZER,
                cache_dir=cache_dir,
            )
            local = Path(ONNX_LOCAL_PATH) if ONNX_LOCAL_PATH else None
            if local is not None and local.is_file():
                onnx_path = str(local)
            else:
                onnx_path = hf_hub_download(
                    repo_id=ONNX_REPO,
                    filename=ONNX_FILENAME,
                    cache_dir=cache_dir,
                )
            self.session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise OnnxEncoderError(f"ONNX MiniLM encoder failed to initialize: {exc}") from exc

        self.input_names = [i.name for i in self.session.get_inputs()]
        self._rag_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("[MEM] ONNX encoder loaded inputs=%s (rss_mb=%s)", self.input_names, process_rss_mb())

    def encode(self, sentences, convert_to_numpy=True, show_progress_bar=False, **_kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        logger.info("[MEM] before ONNX encode n=%s (rss_mb=%s)", len(sentences), process_rss_mb())
        vectors = [self._encode_one(text) for text in sentences]
        logger.info("[MEM] after ONNX encode (rss_mb=%s)", process_rss_mb())
        arr = np.stack(vectors, axis=0)
        logger.info("[MEM] ONNX embedding dim=%s", int(arr.shape[-1]))
        if int(arr.shape[-1]) != 384:
            raise OnnxEncoderError(f"ONNX embedding dim is {arr.shape[-1]}, expected 384")
        return arr if convert_to_numpy else arr.tolist()

    def _encode_one(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text or "",
            padding=True,
            truncation=True,
            max_length=ONNX_MAX_LENGTH,
            return_tensors="np",
        )
        feeds = {}
        for name in self.input_names:
            if name in encoded:
                feeds[name] = encoded[name]
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(encoded["input_ids"])
            else:
                raise OnnxEncoderError(f"ONNX input {name} is not provided by the tokenizer")
        outputs = self.session.run(None, feeds)
        hidden = outputs[0]
        mask = encoded["attention_mask"].astype(np.float32)
        if hidden.ndim != 3:
            raise OnnxEncoderError(f"Unexpected ONNX hidden shape {hidden.shape}")
        mask_exp = mask[:, :, None]
        pooled = (hidden * mask_exp).sum(axis=1) / np.clip(mask_exp.sum(axis=1), 1e-9, None)
        vec = pooled[0]
        # ST module 2 is Normalize; Chroma stored vectors have L2 norm ~ 1.
        denom = float(np.linalg.norm(vec))
        if denom <= 0:
            raise OnnxEncoderError("ONNX embedding has zero norm")
        return (vec / denom).astype(np.float32)


def get_onnx_encoder() -> MiniLMOnnxEncoder:
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        _encoder = MiniLMOnnxEncoder()
        return _encoder
