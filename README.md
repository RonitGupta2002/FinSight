# FinSight India

An agentic research assistant over NSE/BSE cash equities, F&O data, company filings/concalls, and SEBI regulations — built as a 6-week self-directed learning project.

**Order:** tool calling → agent loop → RAG → memory → evals → guardrails + deploy.

This order is deliberate: the agent loop is built before RAG, so orchestration is understood on its own terms before retrieval gets tangled in. By Week 3, RAG slots in as just another tool the agent already knows how to call.

> ⚠️ **Not investment advice.** All F&O/equity outputs are for research and learning purposes only.

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

## Repo structure

```
finsight-india/
├── week1_tool_calling/     # raw tool calling, no framework
├── week2_agent_loop/       # LangGraph ReAct loop
├── week3_rag/               # filings + SEBI regulation retrieval
├── week4_memory/           # persisted watchlist, multi-turn state
├── week5_evals/            # RAGAS + agent-level scoring
├── week6_deploy/           # guardrails, Streamlit app, deployment
├── requirements.txt
├── .env.example
└── .gitignore
```

Each week folder has its own `README.md` with that week's goal, concepts, and how to run it. Start at Week 1.

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

## Progress

- [x] Week 1 — Tool calling
- [ ] Week 2 — Agent loop
- [ ] Week 3 — RAG
- [ ] Week 4 — Memory
- [ ] Week 5 — Evals
- [ ] Week 6 — Guardrails + deploy
- [ ] Week 7 (stretch) — Multi-agent critic
