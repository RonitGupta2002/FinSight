"""
FinSight India — Week 5, Part 3, Segment 1: tool-call correctness scoring.

Two things live here:
1. score_tool_calls() — pure scoring logic. Compares eval_set.py's
   `expected_tools` against what the agent actually called, using multiset
   (Counter) comparison so duplicate expectations (e.g. fund_02 expects
   TWO get_company_overview calls) are handled correctly, not just "was
   this tool name called at all."
2. capture_tool_calls_for_question() — the harness Segment 2 will actually
   use to run eval questions live and capture results. Built now, not run
   yet (zero API cost this segment) — see the note below on why the
   existing agent.py logging isn't directly reusable for this.

Why not just reuse agent_loop/tool_call_log.jsonl directly: TOOL_CALL_LOG is
cleared at the START of every ask() call, and chat()/regression_test()/
stress_test() only write it to a file ONCE at the very end of a whole run —
so the existing log only ever captures the LAST question's calls, not every
question's. capture_tool_calls_for_question() below reads TOOL_CALL_LOG
immediately after each individual ask() call, before the next call clears
it — giving one clean, correctly-scoped record per question.
"""

import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from eval_set import EVAL_SET


def score_tool_calls(expected_tools: list, actual_tool_calls: list) -> dict:
    """Score how well the actual tool calls match what was expected.

    Args:
        expected_tools: list of tool names from eval_set.py, e.g.
            ["get_company_overview", "get_company_overview"] — duplicates
            are meaningful (two different tickers looked up).
        actual_tool_calls: list of dicts shaped like agent.py's
            TOOL_CALL_LOG entries: [{"tool": name, "args": ..., "result": ...}, ...]

    Returns a dict with:
        recall: fraction of expected tool calls that were actually made
                (1.0 if expected_tools is empty and none were called)
        precision: fraction of actual tool calls that were actually expected
                   (1.0 if expected_tools is empty and none were called;
                   0.0 if expected_tools is empty but something WAS called —
                   this is what flags a hallucinated/unnecessary tool call)
        expected / actual / missing / extra: the raw multisets, for
        human-readable failure inspection.
    """
    expected_counter = Counter(expected_tools)
    actual_names = [c["tool"] for c in actual_tool_calls]
    actual_counter = Counter(actual_names)

    matched_counter = expected_counter & actual_counter  # multiset intersection
    total_matched = sum(matched_counter.values())
    total_expected = sum(expected_counter.values())
    total_actual = sum(actual_counter.values())

    if total_expected > 0:
        recall = total_matched / total_expected
    else:
        # Nothing was supposed to be called (e.g. price_05's unanswerable
        # commodity question) — recall is trivially perfect since there's
        # nothing to have missed.
        recall = 1.0

    if total_actual > 0:
        precision = total_matched / total_actual
    else:
        # Nothing was actually called — precision is perfect only if that's
        # also what was expected (nothing). If something WAS expected but
        # zero calls were made, that's a recall failure, not scored here as 0
        # precision (undefined/vacuous), so default to 1.0 and let recall
        # alone carry that failure signal.
        precision = 1.0

    missing_counter = expected_counter - actual_counter
    extra_counter = actual_counter - expected_counter

    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "expected": list(expected_counter.elements()),
        "actual": list(actual_counter.elements()),
        "missing": list(missing_counter.elements()),
        "extra": list(extra_counter.elements()),
    }


def capture_tool_calls_for_question(question: str, thread_id: str) -> list:
    """Run one eval question through the live agent and return exactly the
    tool calls made for THAT question — not built/run in this segment
    (zero API cost so far), used starting Segment 2.

    Import of agent_loop.agent is deliberately deferred inside this function
    (not at module level) — importing it eagerly would try to build LLM
    clients and discover API keys even when this file is only being used
    for the zero-cost scorer logic (as in this segment's self-test below).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent_loop"))
    import agent  # noqa: local import, see docstring

    agent.ask(question, thread_id=thread_id)
    # TOOL_CALL_LOG is cleared at the START of the NEXT ask() call, not this
    # one — so immediately after ask() returns, it still holds exactly this
    # question's calls. Copy it now, before anything else touches it.
    return list(agent.TOOL_CALL_LOG)


def _self_test():
    """Zero-cost validation of score_tool_calls() against the kinds of cases
    actually present in eval_set.py — run this before trusting the scorer
    on real (API-cost-bearing) data in later segments."""
    print("=== Segment 1 self-test: score_tool_calls() ===\n")

    # Case 1: perfect match, single tool (e.g. price_01)
    result = score_tool_calls(
        expected_tools=["get_stock_price"],
        actual_tool_calls=[{"tool": "get_stock_price", "args": {"ticker": "TCS.NS"}, "result": "{}"}],
    )
    print("Case 1 (perfect single match):", result)
    assert result["recall"] == 1.0 and result["precision"] == 1.0

    # Case 2: duplicate expected tools, both made (e.g. fund_02)
    result = score_tool_calls(
        expected_tools=["get_company_overview", "get_company_overview"],
        actual_tool_calls=[
            {"tool": "get_company_overview", "args": {"ticker": "TCS.NS"}, "result": "{}"},
            {"tool": "get_company_overview", "args": {"ticker": "INFY.NS"}, "result": "{}"},
        ],
    )
    print("Case 2 (duplicate expected, both made):", result)
    assert result["recall"] == 1.0 and result["precision"] == 1.0

    # Case 3: missing one of two expected calls (partial recall failure)
    result = score_tool_calls(
        expected_tools=["get_company_overview", "get_company_overview"],
        actual_tool_calls=[
            {"tool": "get_company_overview", "args": {"ticker": "TCS.NS"}, "result": "{}"},
        ],
    )
    print("Case 3 (missing one of two expected):", result)
    assert result["recall"] == 0.5 and result["precision"] == 1.0
    assert result["missing"] == ["get_company_overview"]

    # Case 4: extra unexpected tool called alongside the right one (precision hit)
    result = score_tool_calls(
        expected_tools=["get_stock_price"],
        actual_tool_calls=[
            {"tool": "get_stock_price", "args": {}, "result": "{}"},
            {"tool": "calculate", "args": {}, "result": "{}"},  # unnecessary
        ],
    )
    print("Case 4 (correct call + unnecessary extra):", result)
    assert result["recall"] == 1.0 and abs(result["precision"] - 0.5) < 0.01
    assert result["extra"] == ["calculate"]

    # Case 5: correctly declined (e.g. price_05 — gold price, no tool covers it)
    result = score_tool_calls(expected_tools=[], actual_tool_calls=[])
    print("Case 5 (correctly declined, nothing expected/called):", result)
    assert result["recall"] == 1.0 and result["precision"] == 1.0

    # Case 6: hallucination — nothing was expected, but a tool got called anyway
    result = score_tool_calls(
        expected_tools=[],
        actual_tool_calls=[{"tool": "get_stock_price", "args": {"ticker": "XAU"}, "result": "{}"}],
    )
    print("Case 6 (hallucinated call when none expected):", result)
    assert result["recall"] == 1.0  # nothing was missed
    assert result["precision"] == 0.0  # but what WAS called wasn't warranted
    assert result["extra"] == ["get_stock_price"]

    # Case 7: completely wrong tool used instead of the right one
    result = score_tool_calls(
        expected_tools=["get_option_chain"],
        actual_tool_calls=[{"tool": "get_stock_price", "args": {}, "result": "{}"}],
    )
    print("Case 7 (wrong tool entirely):", result)
    assert result["recall"] == 0.0 and result["precision"] == 0.0

    print("\nAll self-test cases passed.")


if __name__ == "__main__":
    _self_test()

    print("\n=== Coverage check against eval_set.py ===")
    total = len(EVAL_SET)
    with_expected = sum(1 for q in EVAL_SET if q["expected_tools"])
    without_expected = total - with_expected
    print(f"Total questions: {total}")
    print(f"  With non-empty expected_tools: {with_expected}")
    print(f"  With empty expected_tools (decline-gracefully cases): {without_expected}")