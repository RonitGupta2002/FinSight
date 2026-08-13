"""
Diagnostic only — no API calls, no quota cost. Same pattern as
diagnose_retrieval.py, checking filing_04 (TCS workforce) specifically.
Run from evals/: python diagnose_filing_04.py
"""
import sys
sys.path.insert(0, "../rag")
from retrieval import hybrid_search

results = hybrid_search("What did TCS say about its workforce size in the latest quarter?", doc_type="filing")
if not results:
    print("(no results at all)")
for r in results:
    print(f"[{r['company']}] score={r['relevance_score']} source={r['source']}")
    print(f"  {r['text'][:150]}...")
    print()