# FinSight — Tool-Call Correctness Report (Week 5, Part 3)

Scored: 33/33 questions (0 not yet run)


## Overall

- Average recall: **1.000**
- Average precision: **0.701**

## By category

| Category | N | Avg Recall | Avg Precision |
|---|---|---|---|
| filing | 6 | 1.000 | 0.525 |
| fundamentals | 5 | 1.000 | 0.840 |
| multi_hop | 6 | 1.000 | 0.567 |
| option_chain | 5 | 1.000 | 0.767 |
| price | 5 | 1.000 | 0.800 |
| regulation | 6 | 1.000 | 0.756 |

## Questions scoring below 1.0 on recall and/or precision (15)

| ID | Category | Recall | Precision | Missing | Extra | Read |
|---|---|---|---|---|---|---|
| price_05 | price | 1.0 | 0.0 | — | get_stock_price | 🟢 extra rigor, not a real miss |
| fund_05 | fundamentals | 1.0 | 0.2 | — | get_company_overview, get_stock_price, search_filings, search_filings | 🔴 real failure |
| filing_04 | filing | 1.0 | 0.2 | — | search_filings, search_filings, search_filings, search_filings | 🔴 real failure |
| filing_05 | filing | 1.0 | 0.2 | — | search_filings, search_filings, search_filings, search_filings | 🔴 real failure |
| reg_06 | regulation | 1.0 | 0.2 | — | search_regulations, search_regulations, search_regulations, search_regulations | 🟢 extra rigor, not a real miss |
| multi_04 | multi_hop | 1.0 | 0.2 | — | search_regulations, search_regulations, search_regulations, search_regulations | 🟡 partial — facts right, didn't answer the actual question |
| filing_02 | filing | 1.0 | 0.25 | — | search_filings, search_filings, search_filings | 🟢 extra rigor, not a real miss |
| multi_02 | multi_hop | 1.0 | 0.286 | — | search_filings, search_filings, search_filings, search_regulations, search_regulations | 🔴 real failure |
| opt_02 | option_chain | 1.0 | 0.333 | — | calculate, calculate | 🟢 extra rigor, not a real miss |
| reg_05 | regulation | 1.0 | 0.333 | — | search_regulations, search_regulations | 🟢 extra rigor, not a real miss |
| opt_03 | option_chain | 1.0 | 0.5 | — | calculate, calculate | 🟢 extra rigor, not a real miss |
| filing_03 | filing | 1.0 | 0.5 | — | search_filings | 🟢 extra rigor, not a real miss |
| multi_05 | multi_hop | 1.0 | 0.5 | — | view_watchlist, get_company_overview, search_filings, calculate | 🟡 partial — facts right, didn't answer the actual question |
| multi_06 | multi_hop | 1.0 | 0.667 | — | calculate, calculate | 🟢 extra rigor, not a real miss |
| multi_01 | multi_hop | 1.0 | 0.75 | — | calculate | 🟢 extra rigor, not a real miss |

## Real failures (4) — detail

**fund_05** (fundamentals): The Tata Motors ambiguous-ticker case. Model tried TATAMOTORS.NS (404), then guessed TATASTEEL.NS — a WRONG, unrelated company, not just a silent guess within the Tata-Motors ambiguity the question intended. recall=1.0 is misleading here: the one matching get_company_overview call was against the wrong company. Hit the iteration cap; original cap-recovery returned a literal empty answer (bug found and patched — see agent.py notes).

**filing_04** (filing): The correct answer (593,798 workforce) was retrieved TWICE (iterations 3 and 5) but the model kept re-querying instead of stopping, hit the iteration cap before synthesizing, and the original cap-recovery threw the already-correct answer away entirely (same bug as fund_05, now patched).

**filing_05** (filing): The single most important honesty check in the eval set (Q3 vs Q1 FY27 period mismatch). The correct answer (533M subscribers, Q1 FY27) was in the FIRST search result, but 4 more reworded queries followed, hit the cap, and even the patched cap-recovery only produced a generic 'try again' message rather than the expected honest period-mismatch disclosure. The round-1 patch stopped this from being a BLANK answer; it did not make it a GOOD answer. Root cause (over-searching before the cap) is still open.

**multi_02** (multi_hop): Both halves of the answer (533M Jio subscribers; SEBI contract-size regulation at relevance 9.735) were in TOOL_CALL_LOG by iteration 4, but 3 more redundant queries followed, hit the cap, and the round-2 digest-rescue FALSELY claimed no SEBI info was found — the correct regulation text existed but was truncated out of the raw-JSON digest before reaching it. First live case confirming the round-2 rescue mechanism works structurally (produced real text, not a blank/apology) but the digest's blind truncation can still lose real answers. Motivated the round-3 digest patch (JSON-aware, relevance-sorted extraction) — not yet re-tested live.

## Partial cases (2) — facts right, question missed

**multi_04** (multi_hop): The round-2 rescue produced factually CORRECT content this time (contract-size Nov 2024, premium collection Feb 2025, calendar spread removal, intraday monitoring Apr 2025 — no false claims) but never actually answered the question asked, which was specifically whether/how these changes affect the user's own NIFTY weekly-options trading. It listed measures instead of reasoning about the user's situation — the question's notes flag this interpretive step as the actual thing being tested. Facts right, the ask itself missed.

**multi_05** (multi_hop): First live test of the round-4 digest patch (guarantees every tool call at least one slot, fixing round-3's crowding-out bug where a single 5-result search_filings call could fill the whole digest and drop unrelated tool calls entirely). Confirmed working: the rescued answer correctly stated ICICI's NIM (4.36%) AND real NIFTY option-chain OI figures (716,411 calls / 615,044 puts at the 24200 strike) — both facts, no false claims. Still 'partial' rather than clean: it explicitly and honestly says it doesn't have a full comparison/trend analysis between the two, rather than fabricating one. Correct facts, correct honesty about the gap, comparison itself incomplete.

## Extra-rigor cases, not real failures (9)

These score below 1.0 (usually on precision) but the underlying behavior was correct or better than the minimal expected sequence — the scorer can't tell 'unnecessary/wrong extra call' apart from 'extra call that made the answer better.'

- **price_05**: Substituted GOLDBEES.NS as a reasonable gold-price proxy and explicitly disclosed it doesn't have a physical-gold-spot tool. Precision penalized for a tool call that was actually good-faith, disclosed behavior.
- **opt_02**: Computed a real PCR (1.03) via calculate to back up its 'heavy put writing' claim, even though calculate wasn't in expected_tools for this question. More rigorous than the minimal expected sequence.
- **opt_03**: Made 3 calculate calls (two sums + one division) against an expected 1 — but this is the exact rigor the question's notes ask for ('not an eyeballed number'), just executed as more granular steps than expected.
- **filing_02**: 4 search_filings calls, but ended with a correct, well-sourced answer including a historical NIM trend table. Over-searched, not wrong.
- **filing_03**: 2 search_filings calls, correct final answer including domestic/overseas NIM breakdown. Mild over-searching, not a real issue.
- **reg_05**: 3 search_regulations calls, but the final answer correctly synthesized all 6 related SEBI measures (weekly index rationalization, tail risk ELM, calendar spread removal, contract size, premium collection, intraday monitoring) into one well-organized, fully correct answer. Over-searched, not wrong.
- **reg_06**: 5 searches, hit the iteration cap, but the final turn happened to produce real text naturally (no cap-recovery needed). The answer correctly gave a partial, honestly-scoped response citing only what the corpus actually covers for this deliberately broad question — exactly the behavior the question's notes call for.
- **multi_01**: Correct P/E comparison (TCS 16.90 > Infosys 14.79) plus an unrequested but genuinely useful PCR calculation (0.74) via calculate to characterize NIFTY options sentiment. More rigorous than the minimal expected sequence.
- **multi_06**: No cap hit at all — clean 6-call run (search_filings, search_regulations, get_option_chain, 3x calculate) producing a correctly computed PCR (0.78) plus accurate TCS figures and SEBI regulation summary. The 'extra' calculate calls ARE the PCR calculation the question explicitly asked for ('what's NIFTY's put-call ratio telling us'). Also directly engaged the comparative judgment call the question asked for, with reasoning. Best multi-hop answer in the set.
