# FinSight India

An agentic research assistant over NSE/BSE cash equities, F&O data, company filings/concalls, and SEBI regulations — built as a 6-week self-directed learning project on a 100% free-tier stack.

**Build order:** tool calling → agent loop → RAG → memory → evals → guardrails + deploy.

This order is deliberate: the agent loop is built before RAG, so orchestration is understood on its own terms before retrieval gets tangled in. By Week 3, RAG slots in as just another tool the agent already knows how to call.

> ⚠️ **Not investment advice.** All F&O/equity outputs are for research and learning purposes only.

**Status: Weeks 1–5 complete.** Week 6 (guardrails + deploy) is in progress. See [Architecture](#architecture) for what's built and [Results](#results) for eval numbers.

## Stack (100% free tier)

| Layer | Tool |
|---|---|
| LLM | Google Gemini API (primary) + Groq (fast/backup) |
| Orchestration | LangGraph |
| Embeddings | sentence-transformers (local) |
| Vector DB | Chroma (local) |
| Reranker | local cross-encoder |
| Web search | Tavily (free tier) |
| Cash equity data | yfinance (`.NS` / `.BO` tickers) |
| F&O + macro data | jugaad-data / nsepython |
| Filings/concalls | NSE/BSE corporate announcement pages, company IR pages |
| Regulations | SEBI circulars (sebi.gov.in) |
| Eval | RAGAS |
| Deploy | Streamlit Community Cloud / HuggingFace Spaces |

## Architecture

By the end of Week 4 the agent has **10 tools**, orchestrated as a LangGraph ReAct loop with conversation memory and a hardened iteration-cap recovery path:

- **5 market-data tools** (Week 1) — `get_stock_price`, `get_company_overview`, `get_option_chain`, `get_repo_rate`, `calculate`
- **2 RAG tools** (Week 3) — `search_filings(query, company=None)`, `search_regulations(query)`. Hybrid BM25 + embedding retrieval, cross-encoder reranked, over **813 chunks** from 6 companies (TCS, Infosys, Reliance, HDFC Bank, ICICI Bank, Tata Motors) plus SEBI circulars
- **3 watchlist tools** (Week 4) — `add_to_watchlist`, `remove_from_watchlist`, `view_watchlist`. SQLite-persisted, mixes equity and F&O instrument types with domain-aware routing (e.g. "margin" correctly resolves to NIM for banks vs. options margin for NIFTY)

Memory (Week 4) adds session-scoped conversation buffering, a watchlist that survives across process restarts, and running-summary compression for long conversations. The agent loop (Week 2) enforces a hard iteration cap with a multi-round cap-recovery fallback, hardened in Week 5 after live evals surfaced four distinct ways a rushed recovery could lose or misreport already-gathered data.

## Results

**Tool-call correctness** — 33 held-out questions, live-run against real Gemini calls:

| Category | Recall | Precision |
|---|---|---|
| price | 1.000 | 0.800 |
| fundamentals | 1.000 | 0.840 |
| option_chain | 1.000 | 0.767 |
| filing | 1.000 | 0.525 |
| regulation | 1.000 | 0.756 |
| multi_hop | 1.000 | 0.567 |
| **Overall** | **1.000** | **0.701** |

15 of 33 questions scored below 1.0 on precision. Manual review classified these as: **9 good-faith "extra rigor"** (calls that improved the answer), **4 real failures** (all traced to one root cause — over-searching past a sufficient answer, triggering the iteration cap), and **2 partial answers** (facts correctly recovered, question not fully answered). All 4 real failures were diagnosed and fixed across four rounds of patches to the cap-recovery path.

**RAG retrieval quality** — RAGAS context precision/recall, LLM-as-judge:

| Corpus | Context precision | Context recall |
|---|---|---|
| Filings | 0.375 | 0.250 |
| Regulations | 0.769 | 0.800 |

Filings retrieval is the weakest part of the system, confirmed independently by both scoring methods — a candidate area to revisit before Week 6.

## Repo structure

```
finsight-india/
├── tool_calling/          # Week 1 — raw tool calling, no framework
│   └── tools_def.py       # get_stock_price, get_company_overview, get_option_chain, get_repo_rate, calculate
├── agent_loop/             # Week 2, 4 — LangGraph ReAct loop, memory, watchlist
│   ├── agent.py            # StateGraph, checkpointer, cap-recovery, retry/backoff, key rotation
│   ├── langgraph_tools.py  # wraps tool_calling/ + rag/ as LangGraph tool nodes
│   ├── watchlist.py        # SQLite CRUD for the persisted watchlist
│   └── watchlist_tools.py  # add_to_watchlist / remove_from_watchlist / view_watchlist
├── rag/                     # Week 3 — filings + SEBI regulation retrieval
│   ├── ingest.py            # extract → chunk → embed → store in Chroma
│   ├── retrieval.py         # hybrid_search() — BM25 + embeddings + cross-encoder rerank
│   ├── retrieval_tools.py   # thin @tool wrappers around retrieval.py
│   ├── test_retrieval.py
│   └── data/                # source PDFs: one folder per company + Regulations/ (gitignored)
├── evals/                   # Week 5 — RAGAS + tool-call-correctness scoring
│   ├── eval_set.py          # 33 held-out questions across 6 categories
│   ├── ragas_eval.py        # context precision/recall on filings + regulation corpora
│   ├── tool_eval.py         # offline scorer — recall/precision on tool calls
│   ├── run_tool_eval.py     # resumable, quota-aware live batch runner
│   └── aggregate_report.py  # dedupes + classifies results (rigor / partial / real failure)
├── week6_deploy/            # not started — guardrails, Streamlit UI, deployment
├── chroma_db/                # local vector store (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
git clone <your-repo-url>
cd finsight-india
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in your free API keys
```

Free API keys needed:
- **Gemini**: [aistudio.google.com](https://aistudio.google.com) — no card required
- **Groq**: [console.groq.com](https://console.groq.com) — no card required
- **Tavily** (Week 3+): [tavily.com](https://tavily.com) — free tier

Before running Week 3+, populate `rag/data/<company>/` with filing/concall PDFs and `rag/data/Regulations/` with SEBI circulars, then build the vector store:

```bash
cd rag
python ingest.py
```

## Progress

- [x] Week 1 — Tool calling (5 tools, wired to both Gemini and Groq)
- [x] Week 2 — Agent loop (LangGraph ReAct, multi-hop tool chaining)
- [x] Week 3 — RAG (813 chunks, hybrid retrieval + reranking, 7 tools total)
- [x] Week 4 — Memory (session memory, persisted mixed-type watchlist, summarization)
- [x] Week 5 — Evals (33-question eval set, RAGAS + tool-call scoring, 4 bugs found and fixed)
- [ ] Week 6 — Guardrails + deploy
- [ ] Week 7 (stretch) — Multi-agent critic