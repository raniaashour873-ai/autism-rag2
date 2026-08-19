"""
Optional embedding-model comparison.

The serving default remains EMBEDDING_MODEL (all-MiniLM-L6-v2).
This script does NOT swap the production model.

Fair comparison requires rebuilding the Chroma collection with the
candidate model. Querying an MiniLM index with a different query encoder
is not a valid A/B test.

Usage:
  set EVAL_ALT_EMBEDDING_MODEL=all-mpnet-base-v2
  python eval_embedding_models.py
"""

from step5_evaluate import evaluate_embedding_models


if __name__ == "__main__":
    print(
        "Default serving model is unchanged.\n"
        "To compare, set EVAL_ALT_EMBEDDING_MODEL and rebuild the index "
        "with that model in a separate collection before treating scores as comparable.\n"
    )
    evaluate_embedding_models()
