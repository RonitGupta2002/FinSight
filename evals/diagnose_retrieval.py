"""
One-off diagnostic for Week 5 Part 2 — no API calls, just inspects what
hybrid_search() actually returns for the two questions that scored badly.
Run from evals/: python diagnose_retrieval.py
"""
import sys
sys.path.insert(0, "../rag")
from retrieval import hybrid_search


def show(label, query):
    print(f"=== {label} ===")
    results = hybrid_search(query, doc_type="filing")
    if not results:
        print("  (no results at all)")
    for r in results:
        print(f"[{r['company']}] score={r['relevance_score']} source={r['source']}")
        print(f"  {r['text'][:150]}...")
        print()


show("filing_03 (ICICI NIM) — the zero-score one",
     "What was ICICI Bank's net interest margin in the most recent quarter?")

show("filing_01 (Jio subscribers) — the recall=0 one",
     "How many total subscribers does Jio have as of the latest Reliance filing?")