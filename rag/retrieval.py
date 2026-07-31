"""
FinSight India — Week 3, Part 2: hybrid retrieval (BM25 + embedding similarity) + reranking.

This file is pure retrieval logic — no tool wrapping, no LangChain (that's Part 3).
Two things happen when you call hybrid_search():

1. Pull top candidates from two different signals, then merge their scores:
   - BM25 (lexical/keyword): great at exact terms — ticker symbols, specific
     numbers, regulation section numbers — things embeddings can blur together.
   - Embedding similarity (semantic): great at paraphrases and concepts — "Q3
     profit growth" matching a chunk that says "third-quarter earnings rose".
2. Rerank the merged candidates with a cross-encoder — a slower model that reads
   the (query, chunk) pair together rather than comparing precomputed vectors,
   and is meaningfully more accurate. Too slow to run over the whole corpus, so
   it only re-scores the top ~20 candidates the cheap methods already narrowed
   down — this "narrow then rerank" two-stage pattern is standard practice.
"""

import os
from rank_bm25 import BM25Okapi

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"                          # small, fast, free, fully local
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"       # small, free, local

HYBRID_CANDIDATES = 20  # how many candidates BM25+embedding retrieval pulls before reranking
FINAL_RESULTS = 5        # how many chunks actually get returned (at most — see MIN_RELEVANCE_SCORE)
MIN_RELEVANCE_SCORE = -4.5  # cross-encoder score floor below which a match is treated as
                             # "not actually relevant" rather than forced through anyway.
                             # See the comment at the filter site for how this was calibrated.

# Loaded lazily on first real search, not at import time — so importing this
# module doesn't immediately trigger a model download or DB connection.
_client = None
_collection = None
_embedder = None
_reranker = None


def _get_resources():
    global _client, _collection, _embedder, _reranker
    if _embedder is None:
        import chromadb
        from sentence_transformers import SentenceTransformer, CrossEncoder
        _client = chromadb.PersistentClient(path=DB_DIR)
        _collection = _client.get_or_create_collection("finsight_docs")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _collection, _embedder, _reranker


def _load_bm25_index(collection, doc_type: str):
    """Build a BM25 index over just this doc_type's chunks.
    Rebuilt per-query rather than cached — simple and correct; if your corpus
    grows large enough that this is slow, that's a sign to persist it instead.
    """
    results = collection.get(where={"doc_type": doc_type}, include=["documents", "metadatas"])
    documents = results["documents"]
    metadatas = results["metadatas"]
    if not documents:
        return None, [], []
    tokenized = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized), documents, metadatas


def hybrid_search(query: str, doc_type: str, company: str = None) -> list[dict]:
    """Combine BM25 keyword scores with embedding similarity scores, then rerank.

    Args:
        query: natural language search query
        doc_type: 'filing' or 'regulation'
        company: optional case-insensitive substring filter against the company
                 metadata tag (e.g. 'reliance' matches 'Reliance Industries').
                 Ignored for doc_type='regulation', which has no company tag.
    """
    collection, embedder, reranker = _get_resources()

    bm25, documents, metadatas = _load_bm25_index(collection, doc_type)
    if bm25 is None:
        return []

    # --- Semantic side: embedding similarity via Chroma ---
    query_embedding = embedder.encode([query]).tolist()
    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(HYBRID_CANDIDATES, len(documents)),
        where={"doc_type": doc_type},
        include=["documents", "metadatas", "distances"],
    )

    # --- Lexical side: BM25 keyword scores over the same doc_type ---
    bm25_scores = bm25.get_scores(query.lower().split())

    # --- Merge: normalize both score sets to 0-1, then average ---
    candidates = {}
    if vector_results["documents"] and vector_results["documents"][0]:
        max_dist = max(vector_results["distances"][0]) or 1
        for doc, meta, dist in zip(vector_results["documents"][0],
                                     vector_results["metadatas"][0],
                                     vector_results["distances"][0]):
            sem_score = 1 - (dist / max_dist)  # smaller distance = more similar
            candidates[doc] = {"metadata": meta, "sem_score": sem_score, "bm25_score": 0}

    max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1
    for doc, meta, score in zip(documents, metadatas, bm25_scores):
        norm_score = score / max_bm25
        if doc in candidates:
            candidates[doc]["bm25_score"] = norm_score
        elif norm_score > 0:
            candidates[doc] = {"metadata": meta, "sem_score": 0, "bm25_score": norm_score}

    if company:
        # filter by the company metadata tag (set from folder name during ingest),
        # not the filename — more robust, e.g. 'reliance' correctly matches the
        # folder/company name 'Reliance Industries'
        candidates = {d: v for d, v in candidates.items()
                      if company.lower() in v["metadata"].get("company", "").lower()}

    scored = [(doc, (v["sem_score"] + v["bm25_score"]) / 2, v["metadata"])
              for doc, v in candidates.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored[:HYBRID_CANDIDATES]

    if not top_candidates:
        return []

    # --- Rerank: cross-encoder re-scores (query, chunk) pairs directly —
    # slower per-pair but far more accurate than the hybrid score alone ---
    pairs = [[query, doc] for doc, _, _ in top_candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(top_candidates, rerank_scores), key=lambda x: x[1], reverse=True)

    # Filter out weak matches instead of always forcing through FINAL_RESULTS
    # regardless of quality. MIN_RELEVANCE_SCORE is an empirically calibrated
    # cutoff, not a universal constant — cross-encoder scores are unbounded and
    # corpus/query-dependent, so this can't be "score > 0". On this project's
    # actual data: a genuinely relevant regulation query scored as low as -2.12,
    # while a deliberately nonsense query topped out at -6.81. -4.5 sits roughly
    # midway between those two clusters, giving margin on both sides. Revisit
    # this number during week 5 evals once there's a larger, scored query set —
    # this is a reasonable starting point, not a tuned final value.
    passing = [(doc, score, meta) for (doc, _, meta), score in reranked if score >= MIN_RELEVANCE_SCORE]

    return [
        {
            "text": doc,
            "source": meta.get("source_file"),
            "company": meta.get("company"),
            "relevance_score": round(float(score), 3),
        }
        for doc, score, meta in passing[:FINAL_RESULTS]
    ]