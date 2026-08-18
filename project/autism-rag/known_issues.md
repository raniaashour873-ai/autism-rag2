# Known Retrieval Limitations

## Failure Case: Generic terminology overlap
**Query:** "Should biological or genetic tests be used for autism diagnosis?"
**Expected:** chunk_0041 (1.2.11 - explicit recommendation against biological/genetic tests)
**Actual:** Retrieved general assessment sections instead (top result: section 1.2 header)

**Root cause:** The embedding model (all-MiniLM-L6-v2) appears to weight common
terms like "assessment" and "diagnosis" heavily, causing confusion between the
specific recommendation and broader section headers that share vocabulary.

**Precision@3 and Precision@5:** 92.86% (13/14 direct questions)

**Potential fixes (not implemented due to time constraints):**
- Try a larger embedding model (e.g., all-mpnet-base-v2)
- Add hybrid search (BM25 + semantic) to catch exact keyword matches
- Rephrase chunk text to include more explicit negative framing

## Known Limitation: Arabic Query Support
The system's embedding model (all-MiniLM-L6-v2) is primarily English-trained.
Arabic queries result in high retrieval distances (1.5-1.8 vs 0.5-0.8 for
English), causing the system to correctly refuse answering but for the wrong
reason (poor semantic matching, not actual lack of evidence).
Fix (not implemented due to time): switch to a multilingual embedding model
such as paraphrase-multilingual-MiniLM-L12-v2.