"""
FinSight India — Streamlit UI
"""

import os
import sys
import uuid
import json

# --- Make agent_loop, rag, and tool_calling importable regardless of the working directory Streamlit Cloud runs this from (repo root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _sub in ("agent_loop", "rag", "tool_calling"):
    _path = os.path.join(_REPO_ROOT, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st

st.set_page_config(page_title="FinSight India", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        header[data-testid="stHeader"] {
            height: 2.5rem;
        }
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
        }
        .st-key-cash_chatbox, .st-key-fo_chatbox {
            height: calc(100vh - 460px) !important;
            min-height: 250px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Starting the agent...")
def load_agent():
    """Imported once per process (Streamlit caches this across reruns and
    across users) — agent.py's own module-level setup (LLM clients, tool
    list, LangGraph compilation) only runs a single time, not per request."""
    import agent as agent_module
    return agent_module


def load_watchlist():
    try:
        from watchlist import list_watchlist
        return list_watchlist()
    except Exception:
        return None  # sidebar handles None as "unavailable", not an error state


agent = load_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "cash_history" not in st.session_state:
    st.session_state.cash_history = []
if "fo_history" not in st.session_state:
    st.session_state.fo_history = []

st.title("📈 FinSight India")
st.caption(
    "Agentic research assistant for Indian cash equity and F&O markets — "
    "Gemini + LangGraph + hybrid RAG over company filings and SEBI regulations."
)

with st.sidebar:
    st.subheader("📋 Watchlist")
    watchlist_items = load_watchlist()
    if watchlist_items is None:
        st.caption("Watchlist unavailable in this deployment.")
    elif not watchlist_items:
        st.caption("Nothing tracked yet — ask the agent to track a stock or index.")
    else:
        st.caption("Click an instrument to ask about it.")
        for item in watchlist_items:
            label = item.get("label") or item.get("symbol")
            symbol = item.get("symbol")
            itype = item.get("instrument_type")
            if st.button(f"{label} · {itype}", key=f"wl_{symbol}", use_container_width=True):
                if itype == "option":
                    st.session_state.fo_pending_question = (
                        f"Show me the current option chain and OI for {symbol}"
                    )
                else:
                    st.session_state.cash_pending_question = (
                        f"What is the latest price and key fundamentals for {symbol}?"
                    )
                st.rerun()

    st.divider()
    st.caption(
        "⚠️ Research information only — not investment advice.\n\n"
        "Running on a free-tier API quota; heavy use may hit a daily limit "
        "and require waiting for it to reset."
    )
    if st.button("Start a new conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.cash_history = []
        st.session_state.fo_history = []
        st.rerun()

tab_cash, tab_fo = st.tabs(["💹 Cash Equity Research", "📊 F&O / OI View"])


def ask_agent(question: str) -> str:
    """Single call site for the UI -> agent boundary. Handles the two
    failure modes agent.py already surfaces as real exceptions/log entries
    (daily quota, anything else) so the UI degrades to a clear message
    instead of a stack trace."""
    try:
        result = agent.ask(question, thread_id=st.session_state.thread_id)
        last_message = result["messages"][-1]
        return agent.extract_text(last_message.content)
    except agent.DailyQuotaExhausted:
        return (
            "⚠️ The free-tier daily API quota has been used up. This isn't fixed "
            "by retrying — it resets on a roughly 24-hour cycle. Please check back later."
        )
    except Exception as e:
        return f"⚠️ Something went wrong answering that: {e}"


def render_view(view_key: str, history_key: str, placeholder: str, example_prompts: list):
    chat_box = st.container(height=420, border=True, key=f"{view_key}_chatbox")
    with chat_box:
        for role, text in st.session_state[history_key]:
            with st.chat_message(role):
                st.markdown(text)

    cols = st.columns(len(example_prompts))
    clicked_prompt = None
    for col, prompt_text in zip(cols, example_prompts):
        if col.button(prompt_text, key=f"{view_key}_ex_{prompt_text}", use_container_width=True):
            clicked_prompt = prompt_text

    pending_key = f"{view_key}_pending_question"
    pending_prompt = st.session_state.pop(pending_key, None)

    typed = st.chat_input(placeholder, key=f"{view_key}_input")
    question = typed or clicked_prompt or pending_prompt
    if not question:
        return

    st.session_state[history_key].append(("user", question))
    with chat_box:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                answer = ask_agent(question)
            st.markdown(answer)
    st.session_state[history_key].append(("assistant", answer))


with tab_cash:
    st.subheader("Cash Equity Research")
    st.caption("Live prices, fundamentals, company filings/concalls, and SEBI regulation search.")
    render_view(
        view_key="cash",
        history_key="cash_history",
        placeholder="e.g. Is TCS's P/E higher than Infosys's?",
        example_prompts=[
            "TCS current price",
            "Compare TCS and Infosys P/E",
            "What did Reliance's filing say about Jio?",
        ],
    )

with tab_fo:
    st.subheader("F&O / Open Interest View")
    st.caption(
        "Option chains, open interest, and the RBI repo rate — always presented as "
        "research information, never a trading recommendation."
    )
    render_view(
        view_key="fo",
        history_key="fo_history",
        placeholder="e.g. Is there heavy put writing near the current NIFTY strike?",
        example_prompts=[
            "NIFTY option chain",
            "Current RBI repo rate",
            "Any recent SEBI F&O lot-size changes?",
        ],
    )