"""
FinSight India — Week 2: the agent loop.
LangGraph StateGraph that chains week 1's 5 tools across cash-equity and F&O
queries, deciding the order itself, with a hard cap on loop length and
step-by-step logging (reused for evals in week 5).
"""

import sys
import os
import json
import re
import time
from datetime import datetime, timezone
from typing import Annotated, Sequence, TypedDict

# Windows terminals default to cp1252, which can't print ₹ or other non-ASCII
# characters the model may return. Force UTF-8 on stdout so this doesn't crash.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()  # reads the .env file in your project root into os.environ

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_tools import ALL_TOOLS as MARKET_TOOLS

# --- Week 3 Part 3: bring in the RAG tools (search_filings, search_regulations)
# alongside the 5 existing market-data tools ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from retrieval_tools import RAG_TOOLS

# --- Week 4 Part 2: persisted watchlist tools ---
from watchlist_tools import WATCHLIST_TOOLS

ALL_TOOLS = MARKET_TOOLS + RAG_TOOLS + WATCHLIST_TOOLS

MAX_ITERATIONS = 6  # safety cap — without this, a confused model can loop forever
                     # and burn your free-tier quota. Tune this up only if you hit
                     # it on a genuinely multi-hop query, not to paper over a bug.

# Week 4 Part 3: summarization thresholds. A single turn with several tool
# calls can easily be 6-10 messages on its own (human question, tool-call
# request, one ToolMessage per call, final answer) — these numbers are set
# generously with that in mind, not tuned for simple one-shot Q&A.
SUMMARY_TRIGGER_MESSAGES = 20  # summarize once the buffer exceeds this many messages
KEEP_RECENT_MESSAGES = 10       # always keep this many of the MOST RECENT messages verbatim

MAX_RETRIES = 4          # how many times to retry a rate-limited call before giving up
DEFAULT_BACKOFF = 15     # seconds to wait if the API doesn't tell us how long to wait

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# Every tool call this run makes gets logged here — week 5 reuses this shape
# directly to score tool-call correctness against your eval set.
TOOL_CALL_LOG = []


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int
    summary: str  # Week 4 Part 3: running summary of older, trimmed-away messages


MODEL = "gemini-3.5-flash"  # gemini-2.5-flash failed (not available on this account/key —
                             # confirmed via list_models.py / direct testing), so back to
                             # gemini-3.5-flash despite its lower 20/day quota. Key rotation
                             # (QUESTIONS_PER_KEY below) is what actually compensates for that
                             # now, not the model choice.

# Week 5: automatic key rotation. Built from whatever numbered keys actually
# exist in .env (GEMINI_API_KEY1, GEMINI_API_KEY2, GEMINI_API_KEY3, ...) —
# add a third key later and it's picked up automatically, no code change.
# Falls back to the old single-key names if no numbered keys are set at all,
# so weeks 1-4's .env setup still works untouched.
QUESTIONS_PER_KEY = 8  # proactively rotate BEFORE hitting the daily wall, not after —
                        # matches the batch size these eval runs are already sized for.

CALL_LOG_PATH = os.path.join(os.path.dirname(__file__), "api_call_log.jsonl")


def _discover_key_pool() -> list[str]:
    """Find every GEMINI_API_KEYn defined in .env, in order. Falls back to
    the older single-key names if none of the numbered ones are set."""
    pool = []
    i = 1
    while os.environ.get(f"GEMINI_API_KEY{i}"):
        pool.append(f"GEMINI_API_KEY{i}")
        i += 1
    if pool:
        return pool
    fallback = "GOOGLE_API_KEY" if os.environ.get("GOOGLE_API_KEY") else "GEMINI_API_KEY"
    if os.environ.get(fallback):
        return [fallback]
    raise RuntimeError(
        "No API key found. Set GEMINI_API_KEY1 (and optionally GEMINI_API_KEY2, "
        "GEMINI_API_KEY3, ...) in your .env file, or GOOGLE_API_KEY / GEMINI_API_KEY "
        "for the older single-key setup."
    )


KEY_POOL = _discover_key_pool()
_key_index = 0                # which key in KEY_POOL is currently active
_questions_on_current_key = 0  # how many ask() calls have used it so far


def _log_call(purpose: str, key_alias: str, success: bool, error: str = None):
    """Append one line per actual API call — every call, not just failures.
    This is the raw material for week 5's eval scoring: cost/latency tracking
    and tool-call correctness both want to know exactly what was called, when,
    with which key, and whether it succeeded."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,          # "agent" | "no_tools" | "summarize"
        "key_alias": key_alias,
        "success": success,
        "error": error,
    }
    with open(CALL_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _build_llm():
    """(Re)build the LLM clients bound to whichever key is currently active.
    Called once at startup and again every time _rotate_key() switches keys —
    langchain_google_genai binds an api_key at construction time, so a real
    key swap means constructing a new client, not just changing a variable."""
    global llm, llm_with_tools, llm_no_tools
    active_key_name = KEY_POOL[_key_index]
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0,
        api_key=os.environ[active_key_name],
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    llm_no_tools = llm


def _rotate_key():
    """Move to the next key in the pool and rebuild the LLM clients. Raises
    once every key in the pool has been used up for this run — at that point
    waiting for tomorrow (or adding another GEMINI_API_KEYn) is the only option."""
    global _key_index, _questions_on_current_key
    if _key_index + 1 >= len(KEY_POOL):
        raise RuntimeError(
            f"All {len(KEY_POOL)} configured API key(s) have each handled "
            f"{QUESTIONS_PER_KEY} questions this run. Add another GEMINI_API_KEYn "
            f"to .env, or wait for a key's daily quota to reset."
        )
    _key_index += 1
    _questions_on_current_key = 0
    print(f"[key rotation] Switching to {KEY_POOL[_key_index]} after "
          f"{QUESTIONS_PER_KEY} questions on the previous key.")
    _build_llm()


_build_llm()  # initial construction, using KEY_POOL[0]
# Same model, no tools bound — used only as a last resort when the iteration
# cap is hit mid-tool-call, so the model is forced to produce a text answer
# instead of requesting yet another tool call it won't be allowed to make.

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research assistant for Indian cash equity and F&O markets. "
    "You have tools for live market data (prices, fundamentals, option chains, repo rate), "
    "company filings/concall transcripts (search_filings), SEBI regulations (search_regulations), "
    "and a persisted watchlist (add_to_watchlist, remove_from_watchlist, view_watchlist) that "
    "survives across sessions. "
    "Use the tools available to you to answer with real data — don't guess numbers or quote "
    "regulations from memory. If no tool can answer part of a question, say so plainly instead "
    "of inventing data.\n\n"
    "Important: if a search tool returns relevant results, use them — do not repeat the same "
    "search with only minor rewording hoping for a better match. If the user asks about a specific "
    "period (e.g. 'Q3') but the available documents only cover a different period (e.g. 'Q1 FY27'), "
    "use the most recent available data and clearly tell the user which period it actually covers, "
    "rather than searching repeatedly for an exact label match that may not exist in the corpus.\n\n"
    "When the user refers to 'my watchlist', 'the stocks I'm tracking', 'their margins/OI/etc.', "
    "OR asks you to summarize/recap the conversation, call view_watchlist FIRST to see what's "
    "ACTUALLY tracked before answering — don't assume, and don't rely on conversation memory alone. "
    "A stock being discussed or looked up earlier in the conversation does NOT mean it was added to "
    "the watchlist — only add_to_watchlist actually tracks something. Never state that an instrument "
    "is 'on the watchlist' unless view_watchlist actually confirms it. "
    "The watchlist mixes equities and F&O instruments; apply the right kind of follow-up to the "
    "right kind of instrument (e.g. margin/OI questions to options, fundamentals/price to equities), "
    "not the same treatment to everything."
))


def extract_text(content) -> str:
    """Gemini 3.5 sometimes returns content as a list of structured blocks
    (e.g. [{'type': 'text', 'text': '...', 'extras': {...}}]) instead of a
    plain string. Pull just the human-readable text back out for printing.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(content)

def _build_tool_log_digest(tool_log, max_snippets=8, max_chars_per_snippet=300, max_total_chars=3000):
    """Round 4: guarantee every distinct tool call contributes at least one
    snippet before spending remaining budget on the highest-relevance
    leftovers. Round 3 fixed losing data to raw truncation, but a single
    search call returning several individually-scored sub-results could
    still crowd out every OTHER tool call's data entirely (confirmed live
    on multi_05: one 5-result search_filings call filled 5 of 6 digest
    slots, dropping option-chain and calculate results that were also
    needed)."""
    per_call_best = []
    leftover_candidates = []
    for entry in tool_log:
        raw = entry.get("result", "")
        tool_name = entry.get("tool", "?")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            parsed = None
        call_candidates = []
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            for item in parsed["results"]:
                text = item.get("text", "")
                score = item.get("relevance_score", 0)
                if text:
                    call_candidates.append((score, text[:max_chars_per_snippet], tool_name))
        else:
            snippet = str(raw)[:max_chars_per_snippet]
            call_candidates.append((0, snippet, tool_name))
        if not call_candidates:
            continue
        call_candidates.sort(key=lambda c: c[0], reverse=True)
        per_call_best.append(call_candidates[0])        # guaranteed slot per call
        leftover_candidates.extend(call_candidates[1:])  # rest compete for remaining budget

    leftover_candidates.sort(key=lambda c: c[0], reverse=True)
    ordered = per_call_best + leftover_candidates

    parts, total, count = [], 0, 0
    for score, text, tool_name in ordered:
        if count >= max_snippets:
            break
        piece = f"[{tool_name}] {text}"
        if total + len(piece) > max_total_chars:
            continue
        parts.append(piece)
        total += len(piece)
        count += 1
    return "\n\n".join(parts)


def invoke_with_retry(messages):
    """Wraps the LLM call with retry + backoff on rate limits (HTTP 429).
    Free-tier quotas are small (as low as 5 requests/minute on some models) —
    a multi-hop query alone can exhaust it, so this is not optional plumbing,
    it's the difference between the script working and crashing mid-run.
    """
    return _invoke_with_retry(llm_with_tools, messages, purpose="agent")


def invoke_with_retry_no_tools(messages):
    """Same retry/backoff, but against the no-tools model — used only for the
    cap-recovery fallback in ask(), so the model can't request another tool
    call it won't be allowed to make."""
    return _invoke_with_retry(llm_no_tools, messages, purpose="no_tools")


class DailyQuotaExhausted(Exception):
    """Raised when the API reports a per-day (not per-minute) quota is used up.
    Unlike a per-minute rate limit, no amount of waiting within the same day
    fixes this — retrying is actively counterproductive, so this is raised
    immediately instead of going through the backoff loop. With proactive
    rotation (QUESTIONS_PER_KEY) this should be rare in practice — it's the
    safety net for when a key was already partially used before this run
    started, not the primary rotation mechanism."""
    pass


def _invoke_with_retry(model, messages, purpose: str = "agent"):
    key_alias = KEY_POOL[_key_index]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.invoke(messages)
            _log_call(purpose, key_alias, success=True)
            return response
        except ChatGoogleGenerativeAIError as e:
            msg = str(e)

            # A daily quota (RPD) exhaustion looks like a 429 too, but retrying
            # within the same day cannot fix it — Google's daily quotas reset on
            # a rolling ~24h basis, not on a per-minute backoff timescale. Fail
            # fast with a clear message instead of burning retries and time.
            if "PerDay" in msg or "RequestsPerDay" in msg:
                _log_call(purpose, key_alias, success=False, error="daily_quota_exhausted")
                raise DailyQuotaExhausted(
                    f"Daily request quota exhausted for '{MODEL}' on key '{key_alias}'. "
                    "This will NOT be fixed by waiting a few seconds — it resets on "
                    "a ~24h cycle. Either wait, or add another GEMINI_API_KEYn to .env "
                    "so rotation has somewhere to go next run — check current quota "
                    "numbers at https://ai.google.dev/gemini-api/docs/rate-limits."
                ) from e

            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not is_rate_limit or attempt == MAX_RETRIES:
                _log_call(purpose, key_alias, success=False, error=msg[:200])
                raise  # not a rate limit, or we're out of retries — let it surface

            # The API often tells us exactly how long to wait — use that if present,
            # otherwise fall back to a fixed delay.
            match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
            wait_s = int(match.group(1)) + 1 if match else DEFAULT_BACKOFF
            print(f"[rate limited] attempt {attempt}/{MAX_RETRIES}, waiting {wait_s}s before retry...")
            time.sleep(wait_s)


def call_agent(state: AgentState) -> dict:
    """The 'reason' step — the model decides what to do next given the conversation so far."""
    messages = [SYSTEM_PROMPT]
    summary = state.get("summary", "")
    if summary:
        # The running summary stands in for messages that have been trimmed
        # away (see summarize_if_needed) — without this, trimming would mean
        # genuinely forgetting things, not just compressing them.
        messages.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
    messages += list(state["messages"])
    response = invoke_with_retry(messages)
    return {"messages": [response], "iterations": state.get("iterations", 0) + 1}


def call_tools(state: AgentState) -> dict:
    """The 'act' + 'observe' step — execute whatever tools the model just requested,
    log each call, and turn results into ToolMessages the model reads next turn."""
    last_message = state["messages"][-1]
    tool_messages = []

    for call in last_message.tool_calls:
        name, args, call_id = call["name"], call["args"], call["id"]
        tool_fn = TOOL_MAP.get(name)

        if tool_fn is None:
            # Day 5 "break it" case: model hallucinated a tool that doesn't exist.
            # Fail gracefully — tell the model, don't crash the loop.
            result = json.dumps({"error": f"No tool named '{name}' exists."})
        else:
            try:
                result = tool_fn.invoke(args)
            except Exception as e:
                result = json.dumps({"error": f"Tool '{name}' raised an exception: {e}"})

        print(f"[tool call] {name}({args}) -> {result}")
        TOOL_CALL_LOG.append({"tool": name, "args": args, "result": result})
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """The stopping condition. Two ways out: the model stopped asking for tools,
    or we hit the iteration cap — whichever comes first."""
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        print(f"[safety cap] Hit {MAX_ITERATIONS} iterations — stopping to avoid a runaway loop.")
        return "end"

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


# --- Build the graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")  # loop back — this is what makes it an agent, not a one-shot call

# Week 4 Part 1: InMemorySaver gives the graph a conversation buffer — state
# (the messages list) persists across multiple .invoke() calls as long as they
# share the same thread_id. This is SESSION-only memory: it lives in RAM and
# is gone the moment the script exits. Part 2 upgrades this to SqliteSaver,
# which persists to disk and survives between separate script runs.
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)


def summarize_if_needed(thread_id: str):
    """Week 4 Part 3: check this thread's message count, and if it's over the
    threshold, compress everything except the most recent messages into a
    running summary. Call this after a turn completes (see ask() below) —
    not mid-turn, since trimming messages while the agent is still reasoning
    about the current question could remove something it's actively using.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config).values
    messages = state.get("messages", [])

    if len(messages) <= SUMMARY_TRIGGER_MESSAGES:
        return  # nothing to do yet

    to_summarize = messages[:-KEEP_RECENT_MESSAGES]
    to_keep = messages[-KEEP_RECENT_MESSAGES:]
    existing_summary = state.get("summary", "")

    # Build the summarization prompt — explicitly ask the model to preserve
    # concrete facts (tickers, figures, watchlist items) since THOSE are what
    # follow-up questions actually need, not a vague narrative recap.
    convo_text = "\n".join(
        f"{m.__class__.__name__}: {extract_text(m.content)}"
        for m in to_summarize if hasattr(m, "content")
    )
    prompt = (
        "Summarize the following conversation excerpt concisely, but preserve every "
        "concrete fact that a follow-up question might need: specific tickers, prices, "
        "ratios, dates, watchlist items, and any figures mentioned. Do not editorialize "
        "or add commentary — just compress.\n\n"
    )
    if existing_summary:
        prompt += f"Existing summary of even earlier context:\n{existing_summary}\n\n"
    prompt += f"Conversation to fold into the summary:\n{convo_text}"

    print(f"[summarizing] {len(to_summarize)} older messages -> compressed summary "
          f"(keeping last {len(to_keep)} messages verbatim)")
    # Sent as a HumanMessage, not a SystemMessage — Gemini's API requires actual
    # conversation content ('contents') separately from system instructions,
    # and a request containing ONLY a system message has no contents at all,
    # which errors with "contents are required". A HumanMessage always counts
    # as real content.
    response = invoke_with_retry_no_tools([HumanMessage(content=prompt)])
    new_summary = extract_text(response.content)

    # RemoveMessage actually deletes these from the checkpointed state — this
    # is real trimming, not just hiding them from one LLM call.
    remove_ops = [RemoveMessage(id=m.id) for m in to_summarize if hasattr(m, "id") and m.id]
    app.update_state(config, {"messages": remove_ops, "summary": new_summary})


def ask(question: str, thread_id: str = "default"):
    """Ask a question within a conversation thread. Calling this multiple times
    with the SAME thread_id lets the model see the full prior conversation —
    that's the actual Week 4 Part 1 deliverable. Different thread_id = a totally
    separate, unrelated conversation (useful for testing in isolation).

    Note the explicit iterations=0 reset below: 'iterations' is a plain (non-
    reducer) state field, so passing it in the input REPLACES the checkpointed
    value rather than accumulating like 'messages' does. This is deliberate —
    without it, the iteration cap would apply to the ENTIRE conversation's
    total tool calls instead of resetting per turn, and a long conversation
    would hit the cap after just a few turns even if each individual turn only
    needed 1-2 tool calls.

    Week 5: also handles automatic key rotation — every call increments a
    per-key question counter, and rotates to the next configured key BEFORE
    the QUESTIONS_PER_KEY limit is exceeded, rather than waiting for an
    actual quota error.
    """
    print(f"\n=== {question} ===")
    global _questions_on_current_key
    _questions_on_current_key += 1
    if _questions_on_current_key > QUESTIONS_PER_KEY:
        _rotate_key()
        _questions_on_current_key = 1  # this question is the first on the new key
    print(f"[key] using {KEY_POOL[_key_index]} (question {_questions_on_current_key}/{QUESTIONS_PER_KEY} on this key)")

    TOOL_CALL_LOG.clear()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [("user", question)], "iterations": 0}, config)
    last_message = result["messages"][-1]

    # If the iteration cap was hit WHILE the model still wanted to call more
    # tools, the last message has no real answer in it — just an unexecuted
    # tool request. Rather than show the user a blank/empty response, force
    # one more direct call asking the model to answer with whatever it has
    # already gathered, tools disabled this time so it can't request another.
    if getattr(last_message, "tool_calls", None):
        print("[recovering from cap] Forcing a final answer from partial results...")
        wrapup_prompt = SystemMessage(content=(
            "You hit your tool-call limit before finishing. Answer the user's original "
            "question now using ONLY the tool results already gathered in this conversation. "
            "You must respond with plain text and must not attempt to call any tool. "
            "If some part genuinely can't be answered from what you have, say so plainly, "
            "but you must still produce a real text answer — never an empty response."
        ))
        messages = [SYSTEM_PROMPT, wrapup_prompt] + list(result["messages"])
        final_response = invoke_with_retry_no_tools(messages)
        answer_text = extract_text(final_response.content)

        # Observed failure mode (fund_05, filing_04 eval runs): the wrap-up call
        # itself can come back with zero content blocks, which extract_text()
        # renders as the literal string "[]". Fall back to a deterministic,
        # honest message built from the raw tool log instead of showing nothing.
        if not answer_text.strip() or answer_text.strip() == "[]":
            # First synthesis attempt came back empty. Before giving up, try
            # once more with a short, directive prompt built from just the
            # raw tool results already gathered — both fund_05 and
            # filing_04/filing_05 had the real answer sitting in
            # TOOL_CALL_LOG at this exact point; the original wrap-up call
            # just never surfaced it.
            digest = _build_tool_log_digest(TOOL_CALL_LOG)
            if digest:
                print("[recovering from cap] First synthesis was empty — "
                      "retrying with a compact fact digest...")
                digest_prompt = HumanMessage(content=(
                    f"Below are raw results from tools already called while answering: "
                    f"\"{question}\"\n\n{digest}\n\n"
                    "State only the concrete facts above that answer the question, in "
                    "2-3 plain sentences. If nothing above answers it, say so in one sentence."
                ))
                retry_response = invoke_with_retry_no_tools([digest_prompt])
                retry_text = extract_text(retry_response.content)
                if retry_text.strip() and retry_text.strip() != "[]":
                    answer_text = retry_text

        if not answer_text.strip() or answer_text.strip() == "[]":
            tools_tried = ", ".join(sorted({c["tool"] for c in TOOL_CALL_LOG})) or "no tools"
            answer_text = (
                "I gathered some information but hit my reasoning-step limit before "
                f"finishing, and my summary attempts came back empty. Tools I tried: "
                f"{tools_tried}. Please try asking again, ideally about one part of the "
                "question at a time."
            )

        print(f"[final answer] {answer_text}")

        recovered_message = AIMessage(content=answer_text)
        app.update_state(config, {"messages": [recovered_message]})
        summarize_if_needed(thread_id)
        result["messages"] = list(result["messages"]) + [recovered_message]
        return result

    print(f"[final answer] {extract_text(last_message.content)}")
    summarize_if_needed(thread_id)
    return result


def chat():
    """Week 4 Part 1's actual deliverable: an interactive, multi-turn
    conversation. Every question you type shares the same thread_id, so the
    model can see everything said earlier in this session — try asking
    something, then a follow-up using 'it'/'that'/'their' and see if it
    resolves correctly using the conversation buffer.

    Type 'exit' or 'quit' to end. This is SESSION-only memory (Part 1) — once
    you exit, the conversation is gone. Part 2 adds a persisted watchlist that
    survives between runs.
    """
    thread_id = f"session-{int(time.time())}"
    print("FinSight India — interactive mode. Type 'exit' or 'quit' to end.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        try:
            ask(question, thread_id=thread_id)
        except DailyQuotaExhausted as e:
            print(f"\n[STOPPED] {e}")
            break

    with open("tool_call_log.jsonl", "w") as f:
        for entry in TOOL_CALL_LOG:
            f.write(json.dumps(entry) + "\n")


def regression_test():
    """The old scripted test sequence from weeks 2-3 — each question runs in
    its OWN thread_id (see the thread_id= arg below), so they're deliberately
    isolated from each other, unlike chat() where every question shares one
    thread. Useful for confirming nothing regressed, but burns 4 queries'
    worth of daily quota every run — prefer chat() for everyday testing."""
    queries = [
        "Is TCS's P/E higher than Infosys's, and is there unusual options activity "
        "building up in NIFTY this week?",
        "What is the current crude oil price in USD per barrel?",
        "What is HDFC Bank's current stock price?",
        "What did Reliance's Q3 concall say about Jio's subscriber growth, "
        "and has SEBI changed F&O lot-size rules recently?",
    ]

    for i, question in enumerate(queries):
        try:
            ask(question, thread_id=f"regression-test-{i}")
        except DailyQuotaExhausted as e:
            print(f"\n[STOPPED] {e}")
            print(f"[STOPPED] Completed {i}/{len(queries)} queries before hitting the daily cap.")
            break

        if i < len(queries) - 1:
            print("\n[pausing 20s to stay under free-tier rate limits]")
            time.sleep(20)

    with open("tool_call_log.jsonl", "w") as f:
        for entry in TOOL_CALL_LOG:
            f.write(json.dumps(entry) + "\n")


def stress_test():
    """Week 4 Part 3's Day 6-7 deliverable: a 15+ turn conversation, all in ONE
    thread, deliberately mixing equity and F&O references so later turns must
    correctly resolve pronouns/references against a summarized (not fully raw)
    history — this only becomes a real test once summarization has actually
    kicked in partway through, not before.

    WARNING ON QUOTA: 15 turns, several of which need multiple tool calls, can
    easily approach or exceed a 20-req/day free-tier cap in ONE run. If you hit
    DailyQuotaExhausted partway through, that's expected — note which turn
    number it stopped at and resume the rest tomorrow, or switch to a model
    with a higher daily cap (see MODEL near the top of this file) before
    re-running the whole thing in one sitting.
    """
    thread_id = f"stress-test-{int(time.time())}"
    turns = [
        "Track HDFC Bank, ICICI Bank, and NIFTY weekly options",                         # 1
        "What is HDFC Bank's P/E ratio?",                                                  # 2
        "How does that compare to ICICI Bank?",                                            # 3
        # "What is the current NIFTY underlying value?",                                     # 4
        "Is there heavy put writing at any strike near that level?",                       # 5
        "What is TCS's current stock price?",                                              # 6
        # "Add TCS to my watchlist as well",                                                 # 7
        # "What is Infosys's P/E ratio?",                                                    # 8
        # "Which of TCS and Infosys is cheaper on a P/E basis?",                             # 9
        # "What did Reliance's latest filing say about Jio's subscriber numbers?",           # 10
        # "Has SEBI made any recent changes to F&O margin rules?",                          # 11
        # "What's on my watchlist right now?",                                              # 12
        "How are the two bank stocks on my watchlist doing on valuation?",                 # 13
        "And how does the options instrument on my watchlist look for OI trends?",         # 14
        "Summarize everything we've discussed about my watchlist in this conversation.",   # 15
    ]

    for i, question in enumerate(turns, start=1):
        try:
            ask(question, thread_id=thread_id)
        except DailyQuotaExhausted as e:
            print(f"\n[STOPPED] {e}")
            print(f"[STOPPED] Completed {i - 1}/{len(turns)} turns before hitting the daily cap.")
            break

        if i < len(turns):
            print(f"\n[turn {i}/{len(turns)} done, pausing 20s to stay under free-tier rate limits]")
            time.sleep(20)

    with open("tool_call_log.jsonl", "w") as f:
        for entry in TOOL_CALL_LOG:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    if "--regression-test" in sys.argv:
        regression_test()
    elif "--stress-test" in sys.argv:
        stress_test()
    else:
        chat()