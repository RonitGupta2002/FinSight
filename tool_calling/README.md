# Week 1 — Call an LLM API + Tool Calling

**Goal:** Get comfortable with the raw mechanics of function/tool calling, across both cash equity and F&O data sources, before any agent abstraction touches it.

## Concepts covered

- Message format and roles
- Tool/function schemas (JSON schema for parameters)
- How the model "decides" to call a tool vs. respond directly
- Parsing tool-call responses and feeding results back (the round trip)
- Basic F&O vocabulary: option chain, open interest, strike, expiry, lot size

## Files

| File | Purpose |
|---|---|
| `tools.py` | The 5 standalone tools — no LLM involved. Run this first. |
| `wire_gemini.py` | Day 3–5: manual round trip (propose → execute → feed back) using Gemini. |

## The 5 tools

- `get_stock_price(ticker)` — yfinance, `.NS` tickers
- `get_company_overview(ticker)` — fundamentals via yfinance
- `get_option_chain(symbol)` — near-ATM strikes via jugaad-data/nsepython
- `get_repo_rate()` — current RBI repo rate via jugaad-data's RBI module
- `calculate(expression)` — safe AST-based calculator (no `eval()`)

## How to run

```bash
# from repo root, with venv active and requirements installed
cd week1_tool_calling

# Day 1-2: sanity check the tools with no LLM involved
python tools.py

# Day 3-5: wire the first 3 tools to Gemini, manual round trip
python wire_gemini.py
```

## Day-by-day

- **Day 1–2:** Get both API keys (Gemini + Groq), make a first plain-text call to each, understand request/response shape.
- **Day 3–4:** Read each provider's function-calling docs, define schemas for `get_stock_price`, `get_company_overview`, `calculate`. Get the model to *propose* a call — don't execute yet.
- **Day 5:** Wire up execution + round-trip for those 3 tools.
- **Day 6:** Add `get_option_chain` and `get_repo_rate` — same pattern, new data source. Expect NSE/RBI-side flakiness here; this is where retry/backoff habits start mattering.
- **Day 7:** Full end-to-end test of all 5 tools with both Gemini and Groq. Write your own `wire_groq.py` using `wire_gemini.py` as a reference — the response-shape differences between providers *are* the learning objective, not something to skip by copying.

## Output of the week

A working script with 5 tools spanning cash equities, F&O, and macro context, each independently tested end-to-end with two different LLM providers.

## Notes / gotchas

- NSE actively rate-limits/blocks scraping-like traffic — `jugaad-data` builds in caching, but expect occasional failures on `get_option_chain`.
- `jugaad_data.rbi`'s method names have shifted across versions — if `get_repo_rate()` breaks, check the installed version's API before assuming your code is wrong.
- Report figures in ₹ crore/lakh, not raw numbers — `get_company_overview` already converts market cap to crore.

A note for your Week 1 README/notes, since this is genuinely useful to have written down: Google's model lineup is shifting fast right now — 2.5 Flash is being pulled from new-key access, Pro models moved fully behind billing in April, and Gemini 3.x models exist alongside 2.5. If you hit another 404 down the line, the fix is always the same: check ai.google.dev/gemini-api/docs/models for current free-tier eligible models before assuming your code is broken.
