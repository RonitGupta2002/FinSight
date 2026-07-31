# Week 3 — Add RAG

**Goal:** Ground answers in real documents — company filings/concalls and SEBI regulatory text — instead of only live numbers.

This week is split into 3 parts, tackled one at a time:
- **Part 1 — Data collection & chunking** (Day 1–3) ✅ done
- **Part 2 (this README's focus) — Embedding + hybrid retrieval** (Day 4–6)
- **Part 3 — Wrap as tools + wire into the agent** (Day 7) — `search_filings` / `search_regulations`, tested against the week 2 agent

## Folder structure (final, as actually used)

```
rag/data/
├── <Company Name>/*.pdf   -- one subfolder per company, e.g. data/TCS/*.pdf
│                              (company name = folder name, used as metadata)
└── regulations/*.pdf      -- SEBI circulars, flat. Folder name matched
                              case-insensitively, so "Regulations" also works.
```

## Part 1 — Data collection & chunking (recap)

- `extract_text(filepath)` — reads a PDF page by page into one string (pypdf)
- `chunk_text(text)` — word-boundary-safe sliding-window chunker, 800 chars / 100 char overlap
- Two real bugs found and fixed during testing: an infinite loop on space-sparse text (dense numeric tables), and duplicate tail fragments on short/exact-chunk-size text. Both verified with targeted test cases.
- Output: 17 real documents across 6 companies + SEBI regulations, cleanly extracted and chunked.

## Part 2 — Embedding + hybrid retrieval

### Concepts covered
- **Embeddings** — a model (`all-MiniLM-L6-v2`, free, runs fully locally, no API calls) converts each text chunk into a vector of numbers that captures its *meaning*. Chunks with similar meaning end up with similar vectors, even if they don't share exact words.
- **Vector similarity search** — given a query, embed it the same way, then find the stored chunks whose vectors are closest to it (Chroma does this efficiently).
- **Hybrid retrieval** — combining two different signals that catch different things:
  - **BM25** (keyword/lexical) — great at exact terms: ticker symbols, specific numbers, regulation section numbers. Misses paraphrases.
  - **Embedding similarity** (semantic) — great at paraphrases and concepts (e.g. "Q3 profit growth" matching "third-quarter earnings rose"). Can blur past exact terms.
  - Both scores are normalized to 0–1 and averaged to rank candidates.
- **Cross-encoder reranking** — a second, slower model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the actual `(query, chunk)` pair together and re-scores the top ~20 hybrid candidates. This is more accurate than comparing precomputed vectors, but too slow to run over the whole corpus — hence "narrow with cheap methods, then rerank the shortlist," which is standard practice in production RAG systems.
- **Metadata filtering** — since Part 1 tagged every chunk with `doc_type` and `company`, retrieval can restrict a search to one company or one document type without re-scanning everything.

### Resources
- ChromaDB's own cross-encoder reranking cookbook (the exact pattern used here): https://cookbook.chromadb.dev/embeddings/cross-encoders/
- `sentence-transformers` docs: https://www.sbert.net/
- `rank_bm25` (the BM25 implementation used): https://github.com/dorianbrown/rank_bm25
- General hybrid search + reranking reference: https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026

### Files
| File | Purpose |
|---|---|
| `ingest.py` | Part 1's extraction/chunking + Part 2's embedding and Chroma storage (`main()`) |
| `retrieval.py` | **New this part.** Pure hybrid retrieval logic — `hybrid_search(query, doc_type, company=None)`. No tool-wrapping yet (that's Part 3). |
| `test_retrieval.py` | Manual test script — run a few realistic queries and eyeball the results |

### Install
```bash
pip install sentence-transformers chromadb rank_bm25
```
The first run will download the embedding model and reranker model from Hugging Face (a few hundred MB total) — this needs real internet access and only happens once; both are cached locally after that.

### How to run
```bash
cd rag

# 1. Build the vector store (embeds every chunk, stores in chroma_db/)
python ingest.py

# 2. Try some real queries against it
python test_retrieval.py
```

### What to actually check in the output (not just "did it run")
1. Do the top results for each query look relevant to a human, not just superficially keyword-matching?
2. Does the `company` filter correctly exclude other companies?
3. Do filing queries ever leak regulation content, or vice versa? (They shouldn't — `doc_type` keeps them separate.)
4. Does a deliberately unrelated query (e.g. "weather forecast for Mumbai tomorrow") come back with weak scores or nothing, rather than confidently returning irrelevant chunks?

### Bugs found and fixed while building this (verified via a mocked test harness, since embedding models need internet access this sandbox doesn't have)
- **Dead code removed**: an earlier draft computed a `where_filter` for ticker filtering that was never actually used in the Chroma query — the real filtering happened via post-hoc filename matching instead. Cleaned up to filter directly against the `company` metadata field, which is more robust (e.g. `"reliance"` correctly matches the folder/company name `"Reliance Industries"`, rather than depending on the filename containing the ticker).
- **Lazy imports** — `chromadb`/`sentence-transformers` are only imported inside functions that need them, not at module level, so `retrieval.py` (and `ingest.py`) can still be partially used/tested without those packages installed.
- Verified via mocked models (real semantic models can't be downloaded in this sandbox) that: company filtering works correctly, `doc_type` separation between filings and regulations is airtight (no cross-contamination), searches with no matching company return an empty list rather than erroring, and re-running `ingest.py` on the same data does not create duplicate chunks (Chroma `upsert` correctly overwrites by ID).

### Output of Part 2
A local Chroma vector store (`chroma_db/`, gitignored) containing every chunk from Part 1, embedded and searchable via `hybrid_search()` — combining keyword and semantic retrieval with cross-encoder reranking, filterable by document type and company.