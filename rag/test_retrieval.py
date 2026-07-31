"""
FinSight India — Week 3, Part 2: manual test script.

Run this AFTER running `python ingest.py` successfully (i.e. after your real
chroma_db/ has been built from your actual filings + regulations).

This just calls hybrid_search() directly with a few realistic queries so you
can eyeball whether retrieval quality looks right, before Part 3 wraps this
in tools and hands it to the agent.
"""

from retrieval import hybrid_search


def show(label, results):
    print(f"\n--- {label} ---")
    if not results:
        print("  (no results)")
        return
    for r in results:
        company_tag = f"[{r['company']}] " if r.get("company") else ""
        print(f"  {company_tag}score={r['relevance_score']}  source={r['source']}")
        print(f"    {r['text'][:150]}...")


if __name__ == "__main__":
    # Adjust these to match your actual companies/content — the point is to
    # sanity-check retrieval, not to match these exact examples.

    show(
        "Filing search, no company filter",
        hybrid_search("revenue growth this quarter", doc_type="filing"),
    )

    show(
        "Filing search, filtered to one company",
        hybrid_search("earnings this quarter", doc_type="filing", company="TCS"),
    )

    show(
        "Regulation search",
        hybrid_search("lot size margin expiry day", doc_type="regulation"),
    )

    show(
        "A query that should return weak/no matches — sanity check",
        hybrid_search("weather forecast for Mumbai tomorrow", doc_type="filing"),
    )

    print("\nThings to actually look at, not just 'did it run':")
    print("  1. Do the top results for each query actually look relevant to a human?")
    print("  2. Does the company filter correctly exclude other companies?")
    print("  3. Do filing queries ever leak regulation content, or vice versa?")
    print("  4. Does the last (nonsense) query correctly return weak scores or nothing,")
    print("     rather than confidently returning irrelevant chunks?")