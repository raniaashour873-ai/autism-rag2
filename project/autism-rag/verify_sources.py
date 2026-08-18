from step4_retrieval import load_collection
from sentence_transformers import SentenceTransformer

collection = load_collection()
model = SentenceTransformer("all-MiniLM-L6-v2")


def check(question, top_k=3):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    print(f"\nQuestion: {question}")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{meta['document_name'][:20]}...] page {meta['page_number']} | {meta['section_title'][:50]} | dist={dist:.3f}")


check("How should autism be diagnosed in children?")
check("Should chelation be used for adults with autism?")