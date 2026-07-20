"""
FinSight India — Day 3-5: wire tools to Gemini, manual round trip.
pip install google-genai
Set GEMINI_API_KEY as an environment variable (get free key at aistudio.google.com).
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from tools_def import get_stock_price, get_company_overview, calculate

# Windows' default terminal encoding (cp1252) can't print ₹ and other non-ASCII
# characters — this project reports figures in ₹ crore/lakh throughout, so this
# fix matters every week, not just here. Forces stdout/stderr to UTF-8 instead.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# .env lives one folder up (FinSight/.env), not next to this script (FinSight/tool_calling/).
# Building the path from this file's own location means it works no matter what
# directory you run the script from — Code Runner's cwd can be inconsistent.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        f"GEMINI_API_KEY not found. Checked for a .env file at: {ENV_PATH}\n"
        "Make sure that file exists and contains a line like: GEMINI_API_KEY=your_real_key"
    )

client = genai.Client(api_key=api_key)

# Map tool names -> actual Python functions, so you can dispatch by name
TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "get_company_overview": get_company_overview,
    "calculate": calculate,
}

# JSON schema declarations — start with just these 3, add option_chain/repo_rate on Day 6
TOOLS = [
    {
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
    {
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
    {
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
]


MODEL = "gemini-3.5-flash"  # GA + free-tier eligible as of July 2026


def ask(question: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config={"tools": [{"function_declarations": TOOLS}]},
    )

    if response.function_calls:
        call = response.function_calls[0]
        print(f"[model wants to call] {call.name}({dict(call.args)})  id={call.id}")

        # Day 3-4 milestone: stop here, just print the proposed call, don't execute yet.
        # Day 5 milestone: uncomment below to execute + round-trip the result.

        result = TOOL_MAP[call.name](**call.args)
        print(f"[tool result] {result}")

        # Feed the result back so the model can produce a final answer.
        # IMPORTANT for Gemini 3.x: every function_response must carry the same
        # `id` the model gave us in the original function call, or this second
        # call fails to map the result back to the right request. 2.5-era models
        # didn't require this — if you copy this pattern elsewhere, don't drop it.
        follow_up = client.models.generate_content(
            model=MODEL,
            contents=[
                question,
                response.candidates[0].content,  # the model's function-call turn
                {"role": "user", "parts": [{
                    "function_response": {
                        "id": call.id,
                        "name": call.name,
                        "response": {"result": result},
                    }
                }]},
            ],
            config={"tools": [{"function_declarations": TOOLS}]},
        )
        print(f"[final answer] {follow_up.text}")
    else:
        print(f"[direct answer] {response.text}")


if __name__ == "__main__":
    ask("What is TCS's current stock price?")
    ask("What is 2 + 2?")  # sanity check: should this even trigger a tool call?