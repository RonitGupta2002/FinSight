"""
FinSight India — Week 5, Part 3, Segment 2: live batch harness.

Runs eval_set.py questions live through agent.py, one at a time, in an
ISOLATED thread_id per question — deliberately NOT a shared conversation.
Eval questions are meant to be scored independently; sharing a thread would
let the model dodge a tool call by reusing another question's already-fetched
data from conversation memory, which would understate recall failures rather
than reveal them (regression_test() in agent.py uses the same per-query
isolated-thread pattern for the same reason).

For each question:
  1. capture_tool_calls_for_question() (from tool_eval.py) runs it live and
     returns exactly the tool calls made for that question.
  2. score_tool_calls() scores it against eval_set.py's expected_tools.
  3. The result is appended to evals/tool_eval_results.jsonl, one line per
     question — so a batch can be resumed after a quota cutoff without
     re-spending API calls on questions already scored (same resumability
     pattern as Part 2's ragas_eval.py).

Costs real API calls per question run. Nothing here executes on import —
only via `python run_tool_eval.py ...`. Use --list first to verify a filter
selects the questions you expect, at zero cost.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from eval_set import EVAL_SET
from tool_eval import score_tool_calls, capture_tool_calls_for_question

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "tool_eval_results.jsonl")
DEFAULT_SLEEP = 20  # seconds between questions — matches agent.py's own
                     # regression_test()/stress_test() pacing, to stay under
                     # free-tier per-minute limits


def _load_already_scored_ids() -> set:
    """A question counts as already scored only if its stored line has no
    'error' field. A question that errored out mid-batch (e.g. an unexpected
    exception, or a DailyQuotaExhausted cutoff) is NOT treated as done, so
    it's picked up again on the next run instead of silently skipped."""
    if not os.path.exists(RESULTS_PATH):
        return set()
    done = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("error") is None:
                done.add(entry["id"])
    return done


def _append_result(entry: dict):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _select_questions(question_ids=None, categories=None, limit=None, force=False):
    questions = EVAL_SET
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    if categories:
        questions = [q for q in questions if q["category"] in categories]

    if not force:
        already_done = _load_already_scored_ids()
        questions = [q for q in questions if q["id"] not in already_done]

    if limit is not None:
        questions = questions[:limit]

    return questions


def list_matches(question_ids=None, categories=None, limit=None, force=False):
    """Zero-cost preview: show exactly which questions a filter would run,
    without making any API calls. Run this before every live batch."""
    questions = _select_questions(question_ids, categories, limit, force)
    if not questions:
        print("No questions match this filter (or everything matching is "
              "already scored — pass --force to include them anyway).")
        return
    print(f"{len(questions)} question(s) would run:\n")
    for q in questions:
        print(f"  {q['id']:15s} [{q['category']}]  expected_tools={q['expected_tools']}")


def run_batch(question_ids=None, categories=None, limit=None, force=False,
              sleep_s=DEFAULT_SLEEP):
    questions = _select_questions(question_ids, categories, limit, force)

    if not questions:
        print("Nothing to run — all matching questions already scored "
              "(use --force to re-run, or --list to see why nothing matched).")
        return

    print(f"Running {len(questions)} question(s) live: "
          f"{[q['id'] for q in questions]}\n")

    # Deferred import — only needed once we're actually about to make live
    # calls, so --list / a bad filter never requires API keys to be configured.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent_loop"))
    from agent import DailyQuotaExhausted

    completed = 0
    for i, q in enumerate(questions):
        thread_id = f"tooleval-{q['id']}"
        print(f"--- [{i + 1}/{len(questions)}] {q['id']} ({q['category']}) ---")
        try:
            actual_calls = capture_tool_calls_for_question(q["question"], thread_id)
        except DailyQuotaExhausted as e:
            print(f"\n[STOPPED] {e}")
            print(f"[STOPPED] Completed {completed}/{len(questions)} questions "
                  f"this run before hitting the daily cap. Re-run the same "
                  f"command later (or tomorrow) — already-scored questions "
                  f"are skipped automatically.")
            break
        except Exception as e:
            print(f"[ERROR] {q['id']} raised an unexpected exception: {e}")
            _append_result({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected_tools": q["expected_tools"],
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            continue  # one bad question shouldn't take down the whole batch

        scoring = score_tool_calls(q["expected_tools"], actual_calls)
        result_entry = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_tools": q["expected_tools"],
            "actual_tool_calls": actual_calls,
            "recall": scoring["recall"],
            "precision": scoring["precision"],
            "missing": scoring["missing"],
            "extra": scoring["extra"],
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _append_result(result_entry)
        completed += 1
        print(f"  recall={scoring['recall']}  precision={scoring['precision']}  "
              f"missing={scoring['missing']}  extra={scoring['extra']}\n")

        if i < len(questions) - 1:
            print(f"[pausing {sleep_s}s to stay under free-tier rate limits]\n")
            time.sleep(sleep_s)

    print(f"\nBatch done — {completed}/{len(questions)} question(s) scored this run. "
          f"Results in {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=str, default=None,
                         help="Comma-separated eval_set ids, e.g. price_01,price_02")
    parser.add_argument("--category", type=str, default=None,
                         help="Comma-separated categories, e.g. price,fundamentals,option_chain")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap how many questions this run processes")
    parser.add_argument("--force", action="store_true",
                         help="Include/re-run questions even if already successfully scored")
    parser.add_argument("--sleep", type=int, default=DEFAULT_SLEEP,
                         help=f"Seconds to pause between questions (default {DEFAULT_SLEEP})")
    parser.add_argument("--list", action="store_true",
                         help="Zero-cost: show which questions would run, without calling the API")
    args = parser.parse_args()

    question_ids = set(args.ids.split(",")) if args.ids else None
    categories = set(args.category.split(",")) if args.category else None

    if args.list:
        list_matches(question_ids=question_ids, categories=categories,
                     limit=args.limit, force=args.force)
        return

    run_batch(question_ids=question_ids, categories=categories,
              limit=args.limit, force=args.force, sleep_s=args.sleep)


if __name__ == "__main__":
    main()