# Week 2 — Add a Loop = Your First Agent

**Goal:** Turn week 1's one-shot tool calls into a loop that chains multiple tools across both cash equity and F&O domains, without scripting the order.

**Status:** Not started yet.

## Concepts to cover
- The ReAct pattern (Reason → Act → Observe, repeat)
- Stopping conditions (max iterations, "final answer" detection)
- Handling a tool that fails or returns nothing
- State management across loop iterations

## Planned output
A LangGraph agent that autonomously chains tool calls across cash equity and F&O data sources to answer a multi-part question (e.g. "Is TCS's P/E higher than Infosys's, and is there unusual options activity building up in NIFTY this week?"), with logging and a safety cap on loop length.

Full plan lives in the root `README.md` / project brief — this file will be filled in with code, day-by-day notes, and run instructions once Week 2 starts.
