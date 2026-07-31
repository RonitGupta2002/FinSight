"""
FinSight India — Week 3, Part 3: wrap Part 2's hybrid_search() as two tools
for the agent — search_filings and search_regulations.

No retrieval logic lives in this file — it's a thin adapter over retrieval.py,
the same pattern as week2's langgraph_tools.py wrapping week1's plain functions.
This keeps exactly one place (retrieval.py) responsible for how search actually
works, so every fix made there (the relevance threshold, company filtering,
lazy model loading) is automatically inherited here with no duplication.
"""

import json
from langchain_core.tools import tool

from retrieval import hybrid_search


@tool
def search_filings(query: str, company: str = None) -> str:
    """Search company quarterly results and earnings concall transcripts.

    Args:
        query: What to search for, e.g. 'Jio subscriber growth Q3'
        company: Optional — restrict to one company, e.g. 'Reliance' or 'TCS'.
                 Matches against company name case-insensitively (partial match
                 is fine, e.g. 'reliance' matches 'Reliance Industries').
                 Leave blank to search all companies' filings.
    """
    results = hybrid_search(query, doc_type="filing", company=company)
    if not results:
        return json.dumps({
            "error": "No relevant filings found for this query. Either no matching "
                     "documents are ingested for that company, or nothing relevant matched."
        })
    return json.dumps({"results": results})


@tool
def search_regulations(query: str) -> str:
    """Search SEBI circulars and F&O regulations (margin rules, lot-size changes, expiry-day rules).

    Args:
        query: What to search for, e.g. 'NIFTY lot size change 2025'
    """
    results = hybrid_search(query, doc_type="regulation")
    if not results:
        return json.dumps({
            "error": "No relevant regulations found for this query. Either no matching "
                     "circulars are ingested, or nothing relevant matched."
        })
    return json.dumps({"results": results})


RAG_TOOLS = [search_filings, search_regulations]