"""Download the MiniLM ONNX graph into ONNX_CACHE_DIR (build/deploy prefetch)."""

from onnx_encoder import get_onnx_encoder


if __name__ == "__main__":
    encoder = get_onnx_encoder()
    vec = encoder.encode(["prefetch"], convert_to_numpy=True)[0]
    print("onnx prefetch ok dim=", len(vec))
