"""
One-time diagnostic — lists every model actually available to YOUR API key
right now, plus what each one supports. Run this once, then set MODEL in
agent.py to something that actually shows up here — don't guess from blog
posts (they go stale fast, as we just saw with gemini-2.5-flash).
"""

import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Models available to your key, that support generateContent (chat/tool calling):\n")
for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    if "generateContent" in actions:
        print(f"  {model.name}")

print("\nPick one of these for the MODEL constant in agent.py — 'flash' variants are")
print("generally the free-tier-friendly ones; avoid 'pro' unless you know you have quota for it.")