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
from typing import Annotated, Sequence, TypedDict

# Windows terminals default to cp1252, which can't print ₹ or other non-ASCII
# characters the model may return. Force UTF-8 on stdout so this doesn't crash.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()  # reads the .env file in your project root into os.environ

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langgraph_tools import ALL_TOOLS

MAX_ITERATIONS = 6  # safety cap — without this, a confused model can loop forever
                     # and burn your free-tier quota. Tune this up only if you hit
                     # it on a genuinely multi-hop query, not to paper over a bug.

MAX_RETRIES = 4          # how many times to retry a rate-limited call before giving up
DEFAULT_BACKOFF = 15     # seconds to wait if the API doesn't tell us how long to wait

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# Every tool call this run makes gets logged here — week 5 reuses this shape
# directly to score tool-call correctness against your eval set.
TOOL_CALL_LOG = []


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research assistant for Indian cash equity and F&O markets. "
    "Use the tools available to you to answer with real data — don't guess numbers. "
    "If no tool can answer part of a question, say so plainly instead of inventing data."
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


def invoke_with_retry(messages):
    """Wraps the LLM call with retry + backoff on rate limits (HTTP 429).
    Free-tier quotas are small (as low as 5 requests/minute on some models) —
    a multi-hop query alone can exhaust it, so this is not optional plumbing,
    it's the difference between the script working and crashing mid-run.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm_with_tools.invoke(messages)
        except ChatGoogleGenerativeAIError as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not is_rate_limit or attempt == MAX_RETRIES:
                raise  # not a rate limit, or we're out of retries — let it surface

            # The API often tells us exactly how long to wait — use that if present,
            # otherwise fall back to a fixed delay.
            match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", msg)
            wait_s = int(match.group(1)) + 1 if match else DEFAULT_BACKOFF
            print(f"[rate limited] attempt {attempt}/{MAX_RETRIES}, waiting {wait_s}s before retry...")
            time.sleep(wait_s)


def call_agent(state: AgentState) -> dict:
    """The 'reason' step — the model decides what to do next given the conversation so far."""
    messages = [SYSTEM_PROMPT] + list(state["messages"])
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

app = graph.compile()


def ask(question: str):
    print(f"\n=== {question} ===")
    TOOL_CALL_LOG.clear()
    result = app.invoke({"messages": [("user", question)], "iterations": 0})
    print(f"[final answer] {extract_text(result['messages'][-1].content)}")
    return result


if __name__ == "__main__":
    # Target query from the plan — needs 4+ tool calls across cash and F&O data,
    # with the model deciding the order itself
    ask("Is TCS's P/E higher than Infosys's, and is there unusual options activity "
        "building up in NIFTY this week?")

    # Free-tier quotas are tight (as low as 5 req/min on some models) and the query
    # above alone can burn most of it — pause here rather than immediately firing
    # the next query into an already-exhausted quota.
    print("\n[pausing 20s to stay under free-tier rate limits]")
    time.sleep(20)

    # Day 5: deliberately break it — no tool covers this, confirm graceful failure
    # instead of hallucination
    ask("What is the current crude oil price in USD per barrel?")

    print("\n[pausing 20s to stay under free-tier rate limits]")
    time.sleep(20)

    # Simple case: should ideally resolve in one hop, good sanity check on iteration count
    ask("What is HDFC Bank's current stock price?")

    # Dump the log — this is the exact artifact week 5's eval scoring reuses
    with open("tool_call_log.jsonl", "w") as f:
        for entry in TOOL_CALL_LOG:
            f.write(json.dumps(entry) + "\n")