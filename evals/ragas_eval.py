"""
FinSight India — Week 5, Part 2: RAGAS metrics on the filings and regulations corpora.

Scores retrieval quality (NOT full agent generation — that's Part 3) using
two reference-based RAGAS metrics that don't require generating a live answer:

  - Context Precision: of the chunks hybrid_search() actually retrieved, how
    many were genuinely useful for answering the question? (signal-to-noise)
  - Context Recall: of everything actually needed to answer correctly, how
    much did retrieval actually find? (completeness)

Both need an LLM judge (Gemini, via the same key rotation pool agent_loop
uses) but do NOT need a generated answer — only the question, the retrieved
chunks, and eval_set.py's `ground_truth` field. That keeps this part's API
usage to ~2 judge calls per question, not 4+, since no separate "generate an
answer" step is needed.

Only questions with BOTH a non-None ground_truth AND a real factual answer
(not a behavioral/meta description) are scored here — see SCORABLE_IDS below
for exactly which ones and why others are excluded.

Run: python ragas_eval.py [--doc-type filing|regulation] [--key N] [--limit N] [--force]
"""

import sys
import os
import json
import types
from datetime import datetime, timezone

# --- Compatibility shim, applied BEFORE importing ragas ---
# ragas (as of the version available today) has a hardcoded, unconditional
# import of ChatVertexAI from a langchain_community submodule that no longer
# exists in current langchain-community releases (it moved to the separate
# langchain-google-vertexai package). This is a confirmed, currently-open
# upstream bug (not a mistake in this project's setup) — installing an older
# langchain-community to work around it instead breaks langchain-core
# compatibility with langgraph/langchain-google-genai, which this project
# actually needs. Since this project never uses VertexAI at all (Gemini is
# used directly via ChatGoogleGenerativeAI), a harmless stub that's never
# instantiated satisfies the broken import without needing the real package.
def _patch_ragas_vertexai_import():
    fake_chat_models = types.ModuleType("langchain_community.chat_models.vertexai")
    class _StubChatVertexAI:
        pass
    fake_chat_models.ChatVertexAI = _StubChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = fake_chat_models

    fake_llms = types.ModuleType("langchain_community.llms")
    class _StubVertexAI:
        pass
    fake_llms.VertexAI = _StubVertexAI
    sys.modules["langchain_community.llms"] = fake_llms


_patch_ragas_vertexai_import()

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
# Using the ragas.metrics (not ragas.metrics.collections) import path
# deliberately — it prints a deprecation warning but is stable and matches
# what every current tutorial/example uses. The newer collections-based API
# takes a different constructor (InstructorBaseRagasLLM, not a LangChain
# wrapper) and looked actively in-flux when checked — not worth the
# instability for this deliverable. Revisit if this path is actually removed.
from ragas.metrics import ContextPrecision, ContextRecall

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from retrieval import hybrid_search

sys.path.insert(0, os.path.dirname(__file__))
from eval_set import EVAL_SET

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ragas_results.jsonl")

# Only questions where ground_truth is a genuine, retrievable FACT (not a
# meta-observation about corpus coverage, and not a multi-doc-type question).
# filing_05 and multi_02 are deliberately excluded — their "ground truth" is
# about the AGENT's expected behavior (disclosing a Q3/Q1 mismatch), not a
# fact that should appear in retrieved chunks, so context precision/recall
# don't meaningfully apply. filing_06 and reg_06 are excluded because their
# ground_truth is None (never manually verified) — nothing to score against.
SCORABLE_IDS = {
    "filing_01", "filing_02", "filing_03", "filing_04",
    "reg_01", "reg_02", "reg_03", "reg_04", "reg_05",
}


def _discover_key_pool() -> list[str]:
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
    raise RuntimeError("No API key found — set GEMINI_API_KEY1 (or GOOGLE_API_KEY/GEMINI_API_KEY) in .env.")


def _load_already_scored_ids() -> set:
    """Read ragas_results.jsonl and return the set of question ids that
    already have a REAL (non-NaN) score for BOTH metrics from a prior run.
    A question with even one NaN metric is treated as not-yet-scored, since
    that's a real failure worth retrying, not a partial success worth keeping.
    This is what makes re-running after a quota reset only fill in the gaps
    instead of re-burning quota on questions that already succeeded.
    """
    if not os.path.exists(RESULTS_PATH):
        return set()

    best_scores = {}  # id -> (has_valid_precision, has_valid_recall)
    with open(RESULTS_PATH) as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["id"]
            precision_ok = entry.get("context_precision") is not None and entry["context_precision"] == entry["context_precision"]  # NaN != NaN
            recall_ok = entry.get("context_recall") is not None and entry["context_recall"] == entry["context_recall"]
            prev = best_scores.get(qid, (False, False))
            best_scores[qid] = (prev[0] or precision_ok, prev[1] or recall_ok)

    return {qid for qid, (p_ok, r_ok) in best_scores.items() if p_ok and r_ok}


def _prepare_questions(doc_type_filter: str = None, force: bool = False, limit: int = None) -> list[dict]:
    """Filter EVAL_SET down to the scorable questions actually still needing
    a run, in EVAL_SET's own fixed order (so --limit chunks are deterministic
    and reproducible across separate invocations — 'part A' always means the
    same 3 questions, not whatever a set's iteration order happens to give).
    """
    already_scored = set() if force else _load_already_scored_ids()

    questions = [
        q for q in EVAL_SET
        if q["id"] in SCORABLE_IDS
        and q["id"] not in already_scored
        and (doc_type_filter is None or q["category"] == doc_type_filter)
    ]
    skipped = [
        q["id"] for q in EVAL_SET
        if q["id"] in SCORABLE_IDS
        and q["id"] in already_scored
        and (doc_type_filter is None or q["category"] == doc_type_filter)
    ]
    if skipped:
        print(f"Skipping {len(skipped)} already-scored question(s): {skipped}")
        print("(use --force to re-score everything from scratch)\n")

    if limit is not None:
        questions = questions[:limit]

    if not questions:
        raise ValueError(
            f"Nothing left to score for doc_type_filter={doc_type_filter!r} — "
            f"all matching questions already have real scores. Use --force to re-run anyway."
        )
    return questions


def build_dataset(questions: list[dict]) -> tuple[EvaluationDataset, list[dict]]:
    """Run hybrid_search() for each given question, package as a RAGAS
    EvaluationDataset. Returns the dataset plus metadata (id, category) so
    results can be matched back up — RAGAS's own output doesn't carry your
    IDs through automatically.
    """
    samples = []
    metadata = []
    for q in questions:
        print(f"  Retrieving for {q['id']}: {q['question'][:60]}...")
        results = hybrid_search(q["question"], doc_type=q["category"])
        contexts = [r["text"] for r in results]
        if not contexts:
            print(f"    WARNING: hybrid_search returned ZERO chunks for {q['id']} — "
                  f"this will likely score 0 on both metrics, which is a real, meaningful failure, "
                  f"not a scoring artifact.")
        samples.append(SingleTurnSample(
            user_input=q["question"],
            retrieved_contexts=contexts,
            reference=q["ground_truth"],
            response="",  # not needed for context_precision/context_recall
        ))
        metadata.append({"id": q["id"], "category": q["category"], "question": q["question"]})

    return EvaluationDataset(samples=samples), metadata


def run_eval(doc_type_filter: str = None, key_index: int = 0, force: bool = False, limit: int = None):
    key_pool = _discover_key_pool()
    if key_index >= len(key_pool):
        raise ValueError(f"key_index={key_index} out of range — only {len(key_pool)} key(s) configured.")
    key_alias = key_pool[key_index]
    print(f"Using {key_alias} ({key_index + 1}/{len(key_pool)} in pool)\n")

    judge_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0, api_key=os.environ[key_alias])
    evaluator_llm = LangchainLLMWrapper(judge_llm)

    questions = _prepare_questions(doc_type_filter, force=force, limit=limit)
    print(f"This run will score: {[q['id'] for q in questions]}\n")

    print("Building dataset (running hybrid_search for each question)...")
    dataset, metadata = build_dataset(questions)
    print(f"\n{len(metadata)} question(s) to score. Running RAGAS evaluation "
          f"(~{len(metadata) * 2} judge LLM calls expected, run SEQUENTIALLY to respect "
          f"free-tier rate limits — this will take a few minutes, that's expected)...\n")

    result = evaluate(
        dataset=dataset,
        metrics=[ContextPrecision(llm=evaluator_llm), ContextRecall(llm=evaluator_llm)],
        # RAGAS defaults to max_workers=16 — 16 CONCURRENT judge LLM calls at once.
        # That overwhelms the free tier's per-minute rate limit almost immediately
        # (this account has seen limits as low as 5 requests/minute on some Gemini
        # tiers), causing most jobs to time out rather than fail cleanly with a
        # readable 429. Forcing max_workers=1 makes judge calls fully sequential —
        # slower wall-clock time, but each call actually gets a fair shot instead
        # of piling up against the same rate limit simultaneously.
        run_config=RunConfig(max_workers=1, timeout=180),
    )

    df = result.to_pandas()
    print("\n=== Results ===")
    for i, row in df.iterrows():
        meta = metadata[i]
        print(f"[{meta['id']}] ({meta['category']}) "
              f"context_precision={row.get('context_precision', 'n/a'):.3f}  "
              f"context_recall={row.get('context_recall', 'n/a'):.3f}")

    avg_precision = df["context_precision"].mean()
    avg_recall = df["context_recall"].mean()
    print(f"\nAverage context_precision: {avg_precision:.3f}")
    print(f"Average context_recall:    {avg_recall:.3f}")

    # Save full results — Part 3 and the final eval report both reuse this.
    with open(RESULTS_PATH, "a") as f:
        for i, row in df.iterrows():
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "id": metadata[i]["id"],
                "category": metadata[i]["category"],
                "context_precision": float(row.get("context_precision", 0)),
                "context_recall": float(row.get("context_recall", 0)),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"\nResults appended to {RESULTS_PATH}")

    return df


if __name__ == "__main__":
    doc_type_filter = None
    key_index = 0
    limit = None
    force = "--force" in sys.argv
    if "--doc-type" in sys.argv:
        doc_type_filter = sys.argv[sys.argv.index("--doc-type") + 1]
    if "--key" in sys.argv:
        key_index = int(sys.argv[sys.argv.index("--key") + 1])
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    run_eval(doc_type_filter=doc_type_filter, key_index=key_index, force=force, limit=limit)