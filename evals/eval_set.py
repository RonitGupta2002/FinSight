"""
FinSight India — Week 5, Part 1: the eval set.

33 questions spanning all 5 tool types (live price, fundamentals, option
chain/OI, filings search, regulation search) plus multi-hop combinations,
per the original plan. This file defines the data only — no API calls here.
Part 2 (RAGAS) and Part 3 (tool-call correctness scoring) both load this.

Design notes on "expected answers":
- Live data (price, fundamentals, option chain) changes daily, so these
  questions don't have a fixed expected NUMBER. Instead, `expected_tools`
  captures what SHOULD be called, and grading checks "did it call the right
  tool and return a plausible, non-error answer" rather than an exact figure.
- Filings and regulations are grounded in real, ingested PDF documents whose
  content is a static historical snapshot (a Q1 FY27 filing doesn't change),
  so `ground_truth` for these IS a fixed, checkable fact — this is what
  Part 2's RAGAS scoring (faithfulness, context precision/recall) runs against.
- `expected_tools` is the ideal minimal tool sequence, used in Part 3 to score
  tool-call correctness against agent_loop/tool_call_log.jsonl and
  agent_loop/api_call_log.jsonl from actual runs.
"""

EVAL_SET = [
    # ---------------- Category: live price (5) ----------------
    {
        "id": "price_01",
        "category": "price",
        "question": "What is TCS's current stock price?",
        "expected_tools": ["get_stock_price"],
        "ground_truth": None,  # live data, no fixed answer
        "notes": "Simple one-hop lookup. Should resolve in a single tool call.",
    },
    {
        "id": "price_02",
        "category": "price",
        "question": "What is HDFC Bank's current stock price?",
        "expected_tools": ["get_stock_price"],
        "ground_truth": None,
        "notes": "Same pattern, different ticker — checks ticker resolution (HDFCBANK.NS).",
    },
    {
        "id": "price_03",
        "category": "price",
        "question": "What is Reliance Industries trading at right now?",
        "expected_tools": ["get_stock_price"],
        "ground_truth": None,
        "notes": "Tests whether 'Reliance Industries' correctly maps to RELIANCE.NS.",
    },
    {
        "id": "price_04",
        "category": "price",
        "question": "What is the current RBI repo rate?",
        "expected_tools": ["get_repo_rate"],
        "ground_truth": None,
        "notes": "Only question that should trigger get_repo_rate — checks it isn't ignored/mis-routed.",
    },
    {
        "id": "price_05",
        "category": "price",
        "question": "What is the current price of gold in India?",
        "expected_tools": [],
        "ground_truth": None,
        "notes": "Deliberate 'break it' case — no tool covers commodities. Should decline gracefully, "
                 "not hallucinate a price or misuse get_stock_price.",
    },

    # ---------------- Category: fundamentals (5) ----------------
    {
        "id": "fund_01",
        "category": "fundamentals",
        "question": "What is TCS's P/E ratio and market cap?",
        "expected_tools": ["get_company_overview"],
        "ground_truth": None,
        "notes": "Single-company fundamentals lookup.",
    },
    {
        "id": "fund_02",
        "category": "fundamentals",
        "question": "Is TCS's P/E ratio higher than Infosys's?",
        "expected_tools": ["get_company_overview", "get_company_overview"],
        "ground_truth": None,
        "notes": "Two-company comparison — checks it calls the tool twice (once per ticker), not once.",
    },
    {
        "id": "fund_03",
        "category": "fundamentals",
        "question": "What is ICICI Bank's 52-week high and low?",
        "expected_tools": ["get_company_overview"],
        "ground_truth": None,
        "notes": "Checks it reads specific sub-fields from the overview payload, not just P/E.",
    },
    {
        "id": "fund_04",
        "category": "fundamentals",
        "question": "Which is more richly valued on a P/E basis — HDFC Bank or ICICI Bank?",
        "expected_tools": ["get_company_overview", "get_company_overview"],
        "ground_truth": None,
        "notes": "Same shape as fund_02 with different companies — checks consistency, not a one-off.",
    },
    {
        "id": "fund_05",
        "category": "fundamentals",
        "question": "What is Tata Motors' P/E ratio?",
        "expected_tools": ["get_company_overview"],
        "ground_truth": None,
        "notes": "Deliberately ambiguous ticker — Tata Motors split into commercial/passenger vehicle "
                 "entities. Correct behavior is asking which one OR clearly stating an assumption, "
                 "not silently guessing one without flagging the ambiguity.",
    },

    # ---------------- Category: option chain / OI (5) ----------------
    {
        "id": "opt_01",
        "category": "option_chain",
        "question": "What is the current NIFTY underlying value?",
        "expected_tools": ["get_option_chain"],
        "ground_truth": None,
        "notes": "Simple one-hop index lookup.",
    },
    {
        "id": "opt_02",
        "category": "option_chain",
        "question": "Is there heavy put writing building up in NIFTY this week?",
        "expected_tools": ["get_option_chain"],
        "ground_truth": None,
        "notes": "Checks it interprets openInterest across strikes qualitatively, not just dumping raw data.",
    },
    {
        "id": "opt_03",
        "category": "option_chain",
        "question": "What is the put-call ratio for NIFTY right now?",
        "expected_tools": ["get_option_chain", "calculate"],
        "ground_truth": None,
        "notes": "Checks it actually computes PCR (sum Put OI / sum Call OI) via calculate, "
                 "not an approximated/eyeballed number.",
    },
    {
        "id": "opt_04",
        "category": "option_chain",
        "question": "What's the implied volatility skew looking like for BANKNIFTY?",
        "expected_tools": ["get_option_chain"],
        "ground_truth": None,
        "notes": "Different index — checks BANKNIFTY resolves correctly, not just NIFTY.",
    },
    {
        "id": "opt_05",
        "category": "option_chain",
        "question": "Show me the option chain for HDFC Bank.",
        "expected_tools": ["get_option_chain"],
        "ground_truth": None,
        "notes": "Individual stock F&O (not an index) — checks get_option_chain handles both "
                 "index and equity symbols correctly.",
    },

    # ---------------- Category: filings search (6) ----------------
    {
        "id": "filing_01",
        "category": "filing",
        "question": "How many total subscribers does Jio have as of the latest Reliance filing?",
        "expected_tools": ["search_filings"],
        "ground_truth": "Jio's total subscriber base was 533.3 million as of the Q1 FY27 filing "
                        "(quarter ended June 2026), up 7.1% year-on-year from 498.1 million.",
        "notes": "Fact grounded in the actual ingested Q1 FY27 Reliance filing.",
    },
    {
        "id": "filing_02",
        "category": "filing",
        "question": "What was HDFC Bank's net interest margin (NIM) last quarter?",
        "expected_tools": ["search_filings"],
        "ground_truth": "HDFC Bank's standalone Net Interest Margin (NIM) was 3.26% for Q1 FY27.",
        "notes": "Checks it correctly interprets 'margin' as NIM for a bank, not options margin.",
    },
    {
        "id": "filing_03",
        "category": "filing",
        "question": "What was ICICI Bank's net interest margin in the most recent quarter?",
        "expected_tools": ["search_filings"],
        "ground_truth": "ICICI Bank's net interest margin was 4.36% for Q1-2027, up from 4.32% "
                        "in Q4-2026 and 4.34% in Q1-2026.",
        "notes": "Same pattern as filing_02 — checks consistency across companies.",
    },
    {
        "id": "filing_04",
        "category": "filing",
        "question": "What did TCS say about its workforce size in the latest quarter?",
        "expected_tools": ["search_filings"],
        "ground_truth": "TCS's workforce stood at 593,798 at the end of the quarter (Q1 FY27 "
                        "earnings call).",
        "notes": "Tests retrieval from the concall transcript specifically, not the press release.",
    },
    {
        "id": "filing_05",
        "category": "filing",
        "question": "What did Reliance's Q3 concall say about Jio's subscriber growth?",
        "expected_tools": ["search_filings"],
        "ground_truth": "No Q3 filing/concall is actually in the ingested corpus — only Q1 FY27 is "
                        "available. Correct behavior is honestly disclosing this gap and using the "
                        "Q1 FY27 data as the most recent available, NOT presenting Q1 data as if it "
                        "were Q3, and not hallucinating Q3-specific figures.",
        "notes": "Deliberately tests the period-mismatch handling fixed in week 3 — this is the "
                 "single most important honesty check in the whole eval set.",
    },
    {
        "id": "filing_06",
        "category": "filing",
        "question": "What did Infosys's latest filing say about revenue growth?",
        "expected_tools": ["search_filings"],
        "ground_truth": None,  # verify manually against the actual ingested Infosys documents
        "notes": "Company with multiple filing types ingested (auditor's report, financial "
                 "statement, fact sheet, press release) — checks it picks a sensible source "
                 "rather than getting confused by having 4 documents for one company.",
    },

    # ---------------- Category: regulation search (6) ----------------
    {
        "id": "reg_01",
        "category": "regulation",
        "question": "Has SEBI changed the minimum contract size for index derivatives recently?",
        "expected_tools": ["search_regulations"],
        "ground_truth": "Yes — SEBI raised the minimum contract value for index derivatives to "
                        "not less than ₹15 lakhs (up from the previous ₹5-10 lakh range set in "
                        "2015), with lot size calibrated so contract value stays within ₹15-20 "
                        "lakhs. Effective for new contracts introduced after November 20, 2024.",
        "notes": "Core regulatory fact — precise figures should match exactly.",
    },
    {
        "id": "reg_02",
        "category": "regulation",
        "question": "When did SEBI's rule on upfront collection of option premium from buyers take effect?",
        "expected_tools": ["search_regulations"],
        "ground_truth": "Effective February 1, 2025.",
        "notes": "Tests precise date retrieval, not just topic-level matching.",
    },
    {
        "id": "reg_03",
        "category": "regulation",
        "question": "What did SEBI say about calendar spread treatment on expiry day?",
        "expected_tools": ["search_regulations"],
        "ground_truth": "SEBI removed the calendar spread treatment/margin benefit on the expiry "
                        "day for equity index derivatives, effective February 1, 2025.",
        "notes": "Two ingested circulars both touch this topic (the framework circular and the "
                 "dedicated 'Review of Calendar Spread margin benefit' circular) — checks it "
                 "doesn't contradict itself across sources.",
    },
    {
        "id": "reg_04",
        "category": "regulation",
        "question": "Does SEBI require uniform trading and delivery lot sizes for commodity derivatives?",
        "expected_tools": ["search_regulations"],
        "ground_truth": "Yes — per SEBI circular SEBI/HO/CDMRD/DNPMP/CIR/P/2019/023 (Alignment of "
                        "Trading Lot and Delivery Lot size), exchanges must follow a policy of "
                        "uniform trading and delivery lot size for commodity derivatives contracts, "
                        "with exceptions allowed only case-by-case with SEBI approval.",
        "notes": "The oldest/most different-topic circular in the corpus (commodities, not equity "
                 "index) — checks retrieval doesn't just always return the newest/biggest document.",
    },
    {
        "id": "reg_05",
        "category": "regulation",
        "question": "Has SEBI made any changes affecting weekly index options?",
        "expected_tools": ["search_regulations"],
        "ground_truth": "Yes — SEBI rationalized weekly index derivatives products: each exchange "
                        "may now offer weekly-expiry contracts on only one benchmark index, "
                        "effective November 20, 2024.",
        "notes": "Paraphrased query (doesn't say 'lot size' or 'contract size' explicitly) — tests "
                 "semantic retrieval, not just keyword matching.",
    },
    {
        "id": "reg_06",
        "category": "regulation",
        "question": "What is the current margin requirement for equity F&O trading?",
        "expected_tools": ["search_regulations"],
        "ground_truth": None,  # verify manually — checks graceful handling if no single ingested
                                # circular directly answers this broad a question
        "notes": "Deliberately broad/vague question — correct behavior may be a partial answer "
                 "citing what IS in the corpus (tail risk coverage, intraday position monitoring) "
                 "while being honest that a comprehensive margin framework isn't fully covered.",
    },

    # ---------------- Category: multi-hop combinations (6) ----------------
    {
        "id": "multi_01",
        "category": "multi_hop",
        "question": "Is TCS's P/E higher than Infosys's, and is there unusual options activity "
                     "building up in NIFTY this week?",
        "expected_tools": ["get_company_overview", "get_company_overview", "get_option_chain"],
        "ground_truth": None,
        "notes": "The week 2 target query — cash equity + F&O in one question, no filings/regs involved.",
    },
    {
        "id": "multi_02",
        "category": "multi_hop",
        "question": "What did Reliance's Q3 concall say about Jio's subscriber growth, and has "
                     "SEBI changed F&O lot-size rules recently?",
        "expected_tools": ["search_filings", "search_regulations"],
        "ground_truth": "Same period-mismatch honesty requirement as filing_05, PLUS must also "
                        "correctly answer the SEBI lot-size question (same fact as reg_01) — tests "
                        "whether splitting into two sub-questions across two tool types causes either "
                        "half to be handled worse than when asked alone.",
        "notes": "The week 3 target query — the hardest single question in this eval set: two tool "
                 "types (filing + regulation) AND the period-mismatch trap in the same question.",
    },
    {
        "id": "multi_03",
        "category": "multi_hop",
        "question": "How does HDFC Bank's valuation compare to its actual profitability (NIM, RoA) "
                     "from its latest filing?",
        "expected_tools": ["get_company_overview", "search_filings"],
        "ground_truth": None,
        "notes": "Combines a LIVE tool (fundamentals/valuation) with a RAG tool (filing) for the "
                 "SAME company in one question — checks it doesn't only reach for one or the other.",
    },
    {
        "id": "multi_04",
        "category": "multi_hop",
        "question": "If I'm tracking NIFTY weekly options, would the recent SEBI rule changes on "
                     "weekly index products affect what I'm trading?",
        "expected_tools": ["search_regulations"],
        "ground_truth": "Potentially yes — SEBI's rationalization means each exchange can only offer "
                        "weekly-expiry contracts on ONE benchmark index now (effective Nov 20, 2024), "
                        "which could affect which weekly NIFTY products remain available depending on "
                        "which index the exchange chose to keep on a weekly cycle.",
        "notes": "Deliberately open-ended/interpretive — checks it reasons from the regulation to the "
                 "user's specific situation rather than just quoting the circular verbatim.",
    },
    {
        "id": "multi_05",
        "category": "multi_hop",
        "question": "Track ICICI Bank and NIFTY weekly options, then tell me how ICICI's margins "
                     "compare to the options' OI trends.",
        "expected_tools": ["add_to_watchlist", "add_to_watchlist", "search_filings", "get_option_chain"],
        "ground_truth": None,
        "notes": "Combines watchlist tools with mixed-type follow-up in a SINGLE question (rather "
                 "than across turns like week 4's stress test) — checks the type-routing logic "
                 "(margin->equity filing, OI->options) works even without the multi-turn setup.",
    },
    {
        "id": "multi_06",
        "category": "multi_hop",
        "question": "Between TCS's latest quarterly performance and the current SEBI stance on "
                     "expiry-day risk management, which feels more relevant to a retail F&O trader "
                     "this week — and what's NIFTY's put-call ratio telling us?",
        "expected_tools": ["search_filings", "search_regulations", "get_option_chain", "calculate"],
        "ground_truth": None,
        "notes": "Deliberately messy/compound question spanning THREE tool types in one ask — the "
                 "hardest tool-selection test in the set. A reasonable answer might legitimately "
                 "push back on the premise ('TCS results aren't really relevant to F&O risk') "
                 "rather than forcing a false comparison — that's an acceptable answer, not a failure.",
    },
]


def summary():
    from collections import Counter
    categories = Counter(q["category"] for q in EVAL_SET)
    print(f"Total questions: {len(EVAL_SET)}")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    ids = [q["id"] for q in EVAL_SET]
    duplicates = [i for i in ids if ids.count(i) > 1]
    if duplicates:
        print(f"WARNING: duplicate ids found: {set(duplicates)}")
    else:
        print("All ids unique.")


if __name__ == "__main__":
    summary()