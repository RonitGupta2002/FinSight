"""
FinSight India — Day 7: wire tools to Groq, manual round trip.
pip install groq
Set GROQ_API_KEY as an environment variable (get free key at console.groq.com).

This mirrors wire_gemini.py deliberately — same 3 tools, same questions —
so you can diff the two files and see exactly where the provider shapes diverge.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from tools_def import get_stock_price, get_company_overview, calculate

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        f"GROQ_API_KEY not found. Checked for a .env file at: {ENV_PATH}\n"
        "Make sure that file exists and contains a line like: GROQ_API_KEY=your_real_key"
    )
client = Groq(api_key=api_key)

# As of mid-2026, Groq deprecated llama-3.1-8b-instant / llama-3.3-70b-versatile.
# openai/gpt-oss-20b is their current recommended free-tier tool-calling model —
# check console.groq.com/docs/models if this has moved on again by the time you read this.
MODEL = "openai/gpt-oss-20b"

TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "get_company_overview": get_company_overview,
    "calculate": calculate,
}

# --- Difference #1 from Gemini ---
# Groq/OpenAI-style tool schemas wrap each declaration in a {"type": "function", "function": {...}}
# envelope. Gemini took a flat list of declarations. Same information, different nesting.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the latest closing price for an NSE-listed stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "NSE ticker with .NS suffix, e.g. TCS.NS"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_overview",
            "description": "Get basic fundamentals (PE ratio, market cap, sector) for an NSE-listed stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "NSE ticker with .NS suffix, e.g. TCS.NS"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. (120.5 - 98.2) / 98.2 * 100"}
                },
                "required": ["expression"],
            },
        },
    },
]


def ask(question: str):
    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    # --- Difference #2 from Gemini ---
    # Gemini gives you response.function_calls directly. Groq/OpenAI nests it under
    # response.choices[0].message.tool_calls, and each call's arguments arrive as a
    # JSON *string* you must json.loads() yourself — Gemini hands you a dict already.
    message = response.choices[0].message
    tool_calls = message.tool_calls

    if tool_calls:
        call = tool_calls[0]
        args = json.loads(call.function.arguments)
        print(f"[model wants to call] {call.function.name}({args})")

        result = TOOL_MAP[call.function.name](**args)
        print(f"[tool result] {result}")

        # --- Difference #3 from Gemini ---
        # The round trip here is OpenAI-style: append the assistant's tool-call message,
        # then a separate {"role": "tool", ...} message carrying the result, tagged with
        # tool_call_id so the model knows which call it answers. Gemini used a
        # function_response part inside a user turn instead — no tool role, no call ID.
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })

        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        print(f"[final answer] {follow_up.choices[0].message.content}")
    else:
        print(f"[direct answer] {message.content}")


if __name__ == "__main__":
    ask("What is TCS's current stock price?")
    ask("What is 2 + 2?")  # same sanity check as wire_gemini.py — compare: does Groq also reach for the tool?