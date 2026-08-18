"""
FinSight India — Week 5, Part 3, Segment 5: aggregate tool-call correctness report.

Reads evals/tool_eval_results.jsonl (written by run_tool_eval.py) and produces:
  - overall + per-category recall/precision averages
  - a list of every question that scored below 1.0 on either metric
  - qualitative annotations for known special cases, because the scorer only
    compares tool NAMES (see score_tool_calls in tool_eval.py) and can't tell
    "extra rigor that produced a better answer" apart from "a real miss," or
    "matched tool name but wrong argument" apart from "matched correctly."
    Those annotations were built from actually reading each answer's text
    during live runs (Segments 3-4) — the raw numbers alone would mislead.

Zero API cost. Run any time after tool_eval_results.jsonl has data in it —
works fine on a partial run, and re-run again once more results land.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from eval_set import EVAL_SET

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "tool_eval_results.jsonl")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "tool_eval_report.md")

# Qualitative annotations discovered while reading actual answer text during
# live Segment 3/4 runs. The scorer can't see these on its own:
#   - "rigor": the model called an extra tool beyond expected_tools, but the
#     extra call made the answer MORE correct/grounded, not less (e.g.
#     computing a real PCR instead of eyeballing one). Precision looks bad;
#     the actual behavior was good.
#   - "real_failure": a genuine problem the raw score doesn't fully capture
#     on its own (wrong argument despite a "correct" tool name, or an answer
#     that never actually got produced).
NOTES = {
    "price_05": ("rigor", "Substituted GOLDBEES.NS as a reasonable gold-price proxy and "
                 "explicitly disclosed it doesn't have a physical-gold-spot tool. Precision "
                 "penalized for a tool call that was actually good-faith, disclosed behavior."),
    "opt_02": ("rigor", "Computed a real PCR (1.03) via calculate to back up its 'heavy put "
               "writing' claim, even though calculate wasn't in expected_tools for this "
               "question. More rigorous than the minimal expected sequence."),
    "opt_03": ("rigor", "Made 3 calculate calls (two sums + one division) against an expected "
               "1 — but this is the exact rigor the question's notes ask for ('not an "
               "eyeballed number'), just executed as more granular steps than expected."),
    "filing_02": ("rigor", "4 search_filings calls, but ended with a correct, well-sourced "
                  "answer including a historical NIM trend table. Over-searched, not wrong."),
    "filing_03": ("rigor", "2 search_filings calls, correct final answer including domestic/"
                  "overseas NIM breakdown. Mild over-searching, not a real issue."),
    "fund_05": ("real_failure", "The Tata Motors ambiguous-ticker case. Model tried "
                "TATAMOTORS.NS (404), then guessed TATASTEEL.NS — a WRONG, unrelated company, "
                "not just a silent guess within the Tata-Motors ambiguity the question intended. "
                "recall=1.0 is misleading here: the one matching get_company_overview call was "
                "against the wrong company. Hit the iteration cap; original cap-recovery "
                "returned a literal empty answer (bug found and patched — see agent.py notes)."),
    "filing_04": ("real_failure", "The correct answer (593,798 workforce) was retrieved TWICE "
                  "(iterations 3 and 5) but the model kept re-querying instead of stopping, hit "
                  "the iteration cap before synthesizing, and the original cap-recovery threw "
                  "the already-correct answer away entirely (same bug as fund_05, now patched)."),
    "filing_05": ("real_failure", "The single most important honesty check in the eval set (Q3 "
                  "vs Q1 FY27 period mismatch). The correct answer (533M subscribers, Q1 FY27) "
                  "was in the FIRST search result, but 4 more reworded queries followed, hit the "
                  "cap, and even the patched cap-recovery only produced a generic 'try again' "
                  "message rather than the expected honest period-mismatch disclosure. The "
                  "round-1 patch stopped this from being a BLANK answer; it did not make it a "
                  "GOOD answer. Root cause (over-searching before the cap) is still open."),
    "reg_05": ("rigor", "3 search_regulations calls, but the final answer correctly synthesized "
               "all 6 related SEBI measures (weekly index rationalization, tail risk ELM, "
               "calendar spread removal, contract size, premium collection, intraday monitoring) "
               "into one well-organized, fully correct answer. Over-searched, not wrong."),
    "reg_06": ("rigor", "5 searches, hit the iteration cap, but the final turn happened to "
               "produce real text naturally (no cap-recovery needed). The answer correctly "
               "gave a partial, honestly-scoped response citing only what the corpus actually "
               "covers for this deliberately broad question — exactly the behavior the "
               "question's notes call for."),
    "multi_01": ("rigor", "Correct P/E comparison (TCS 16.90 > Infosys 14.79) plus an "
                 "unrequested but genuinely useful PCR calculation (0.74) via calculate to "
                 "characterize NIFTY options sentiment. More rigorous than the minimal expected "
                 "sequence."),
    "multi_02": ("real_failure", "Both halves of the answer (533M Jio subscribers; SEBI "
                 "contract-size regulation at relevance 9.735) were in TOOL_CALL_LOG by "
                 "iteration 4, but 3 more redundant queries followed, hit the cap, and the "
                 "round-2 digest-rescue FALSELY claimed no SEBI info was found — the correct "
                 "regulation text existed but was truncated out of the raw-JSON digest before "
                 "reaching it. First live case confirming the round-2 rescue mechanism works "
                 "structurally (produced real text, not a blank/apology) but the digest's blind "
                 "truncation can still lose real answers. Motivated the round-3 digest patch "
                 "(JSON-aware, relevance-sorted extraction) — not yet re-tested live."),
    "multi_03": ("real_failure", "By iteration 5 the agent had PE (15.94), RoA (1.85%), NIM "
                 "(3.26%), current price (728.65), book value (394), and a computed P/B ratio "
                 "(1.85) — genuinely enough to answer well. Hit the cap anyway, and the round-2 "
                 "digest-rescue claimed 'no information' despite RoA appearing within the first "
                 "~150 characters of one search result (well inside the 400-char truncation "
                 "window) and the other results being small enough to fit easily. Unlike "
                 "multi_02, this doesn't look like a pure truncation issue — the rescue call can "
                 "apparently misjudge a digest full of real numbers as empty, especially when "
                 "presented as unlabeled raw JSON. Needs live re-test after the round-3 patch."),
    "multi_03": ("resolved", "PRE-ROUND-3: rescue claimed 'no information' despite having PE "
                 "(15.94), RoA (1.85%), NIM (3.26%), price, book value, and a computed P/B ratio "
                 "in TOOL_CALL_LOG. Re-run after the round-3 digest patch (JSON-aware, "
                 "relevance-sorted extraction) now scores 1.0/1.0 with a genuinely excellent "
                 "answer (historical context, comparison table, defensible conclusion) — the "
                 "cap wasn't even hit on re-run. Confirms round-3 fixed this exact case. This "
                 "entry is kept for history; it no longer appears in the imperfect-scores table."),
    "multi_04": ("partial", "The round-2 rescue produced factually CORRECT content this time "
                 "(contract-size Nov 2024, premium collection Feb 2025, calendar spread removal, "
                 "intraday monitoring Apr 2025 — no false claims) but never actually answered "
                 "the question asked, which was specifically whether/how these changes affect "
                 "the user's own NIFTY weekly-options trading. It listed measures instead of "
                 "reasoning about the user's situation — the question's notes flag this "
                 "interpretive step as the actual thing being tested. Facts right, the ask "
                 "itself missed."),
    "multi_05": ("partial", "First live test of the round-4 digest patch (guarantees every tool "
                 "call at least one slot, fixing round-3's crowding-out bug where a single "
                 "5-result search_filings call could fill the whole digest and drop unrelated "
                 "tool calls entirely). Confirmed working: the rescued answer correctly stated "
                 "ICICI's NIM (4.36%) AND real NIFTY option-chain OI figures (716,411 calls / "
                 "615,044 puts at the 24200 strike) — both facts, no false claims. Still "
                 "'partial' rather than clean: it explicitly and honestly says it doesn't have "
                 "a full comparison/trend analysis between the two, rather than fabricating one. "
                 "Correct facts, correct honesty about the gap, comparison itself incomplete."),
    "multi_06": ("rigor", "No cap hit at all — clean 6-call run (search_filings, "
                 "search_regulations, get_option_chain, 3x calculate) producing a correctly "
                 "computed PCR (0.78) plus accurate TCS figures and SEBI regulation summary. "
                 "The 'extra' calculate calls ARE the PCR calculation the question explicitly "
                 "asked for ('what's NIFTY's put-call ratio telling us'). Also directly engaged "
                 "the comparative judgment call the question asked for, with reasoning. Best "
                 "multi-hop answer in the set."),
}


def load_results():
    """Returns the latest successful result per question id. The results
    file is append-only, so a --force re-run (e.g. multi_03 after the
    round-3 agent.py patch) adds a NEW line rather than replacing the old
    one — without deduping here, a re-scored question would be counted
    twice, once with its stale score and once with its current one,
    corrupting that category's average."""
    if not os.path.exists(RESULTS_PATH):
        return []
    latest_by_id = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("error") is None:
                latest_by_id[entry["id"]] = entry  # later lines overwrite earlier ones
    return list(latest_by_id.values())


def aggregate():
    results = load_results()
    all_ids = {q["id"]: q["category"] for q in EVAL_SET}
    scored_ids = {r["id"] for r in results}
    missing_ids = sorted(set(all_ids) - scored_ids, key=lambda i: (all_ids[i], i))

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    lines = []
    lines.append("# FinSight — Tool-Call Correctness Report (Week 5, Part 3)\n")
    lines.append(f"Scored: {len(results)}/{len(all_ids)} questions "
                 f"({len(missing_ids)} not yet run)\n")

    if missing_ids:
        lines.append(f"**Not yet scored:** {', '.join(missing_ids)}\n")

    # --- overall ---
    if results:
        avg_recall = sum(r["recall"] for r in results) / len(results)
        avg_precision = sum(r["precision"] for r in results) / len(results)
        lines.append(f"\n## Overall\n")
        lines.append(f"- Average recall: **{avg_recall:.3f}**")
        lines.append(f"- Average precision: **{avg_precision:.3f}**\n")

    # --- per category ---
    lines.append("## By category\n")
    lines.append("| Category | N | Avg Recall | Avg Precision |")
    lines.append("|---|---|---|---|")
    for cat in sorted(by_category):
        rows = by_category[cat]
        r_avg = sum(r["recall"] for r in rows) / len(rows)
        p_avg = sum(r["precision"] for r in rows) / len(rows)
        lines.append(f"| {cat} | {len(rows)} | {r_avg:.3f} | {p_avg:.3f} |")
    lines.append("")

    # --- questions below 1.0 on either metric ---
    imperfect = [r for r in results if r["recall"] < 1.0 or r["precision"] < 1.0]
    imperfect.sort(key=lambda r: (r["recall"], r["precision"]))
    lines.append(f"## Questions scoring below 1.0 on recall and/or precision ({len(imperfect)})\n")
    lines.append("| ID | Category | Recall | Precision | Missing | Extra | Read |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in imperfect:
        tag, note = NOTES.get(r["id"], (None, ""))
        read = {"rigor": "🟢 extra rigor, not a real miss",
                "real_failure": "🔴 real failure",
                "partial": "🟡 partial — facts right, didn't answer the actual question",
                "resolved": "✅ resolved by a later patch, re-run confirmed fixed"
                }.get(tag, "⚪ unreviewed — read the answer text")
        missing = ", ".join(r["missing"]) or "—"
        extra = ", ".join(r["extra"]) or "—"
        lines.append(f"| {r['id']} | {r['category']} | {r['recall']} | {r['precision']} | "
                     f"{missing} | {extra} | {read} |")
    lines.append("")

    # --- real failures called out in detail ---
    real_failures = [r for r in results if NOTES.get(r["id"], (None,))[0] == "real_failure"]
    if real_failures:
        lines.append(f"## Real failures ({len(real_failures)}) — detail\n")
        for r in real_failures:
            _, note = NOTES[r["id"]]
            lines.append(f"**{r['id']}** ({r['category']}): {note}\n")

    # --- partial: facts rescued correctly, but didn't answer the actual question ---
    partial_cases = [r for r in results if NOTES.get(r["id"], (None,))[0] == "partial"]
    if partial_cases:
        lines.append(f"## Partial cases ({len(partial_cases)}) — facts right, question missed\n")
        for r in partial_cases:
            _, note = NOTES[r["id"]]
            lines.append(f"**{r['id']}** ({r['category']}): {note}\n")

    # --- rigor cases called out, briefer ---
    rigor_cases = [r for r in results if NOTES.get(r["id"], (None,))[0] == "rigor"]
    if rigor_cases:
        lines.append(f"## Extra-rigor cases, not real failures ({len(rigor_cases)})\n")
        lines.append("These score below 1.0 (usually on precision) but the underlying "
                     "behavior was correct or better than the minimal expected sequence — "
                     "the scorer can't tell 'unnecessary/wrong extra call' apart from "
                     "'extra call that made the answer better.'\n")
        for r in rigor_cases:
            _, note = NOTES[r["id"]]
            lines.append(f"- **{r['id']}**: {note}")
        lines.append("")

    # --- unreviewed imperfect scores (flag for manual read) ---
    unreviewed = [r for r in imperfect if r["id"] not in NOTES]
    if unreviewed:
        lines.append(f"## Unreviewed — read these answers manually ({len(unreviewed)})\n")
        for r in unreviewed:
            lines.append(f"- **{r['id']}** ({r['category']}): recall={r['recall']} "
                         f"precision={r['precision']}")
        lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    report = aggregate()
    print(report)
    print(f"\n(also written to {REPORT_PATH})")