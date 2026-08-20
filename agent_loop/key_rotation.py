"""
Zero-cost test for Item 4 (key rotation persistence + reactive rotation).

No real Gemini calls happen in this script. Test 3/4 replace agent._ask_impl
with a fake function, so agent.ask()'s wrapper logic is tested in isolation
from the real prompt/tool-call logic inside _ask_impl. Test 1/2 test only
the file-persistence functions directly. Constructing an LLM client during
_rotate_key() (via _build_llm()) does not itself make a network call, so
this stays zero-cost even though it needs a valid-looking key in .env.

Run from agent_loop/:
    python test_key_rotation.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
import agent

STATE_FILE = agent.KEY_STATE_PATH


def test_persistence_across_simulated_runs():
    print("=== Test 1: persistence across simulated runs ===")
    agent._key_index = min(1, len(agent.KEY_POOL) - 1)
    agent._questions_on_current_key = 3
    agent._save_key_state()

    loaded_index, loaded_count = agent._load_key_state()
    print(f"Saved:  key_index={agent._key_index}, questions={agent._questions_on_current_key}")
    print(f"Loaded: key_index={loaded_index}, questions={loaded_count}")
    assert loaded_index == agent._key_index
    assert loaded_count == agent._questions_on_current_key
    print("PASS — state survives a simulated fresh run\n")


def test_stale_index_discarded():
    print("=== Test 2: out-of-range saved index is discarded, not trusted ===")
    with open(STATE_FILE, "w") as f:
        json.dump({"key_index": 999, "questions_on_current_key": 5}, f)
    idx, cnt = agent._load_key_state()
    assert idx == 0 and cnt == 0
    print("PASS — safely resets to (0, 0) instead of crashing or using a bad index\n")


def test_reactive_rotation():
    print("=== Test 3: reactive rotation on DailyQuotaExhausted ===")
    if len(agent.KEY_POOL) <= 1:
        print("SKIPPED — only 1 key configured, can't test rotation to a 'next' key\n")
        return

    call_count = {"n": 0}

    def fake_ask_impl(question, thread_id="default"):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise agent.DailyQuotaExhausted("simulated quota exhaustion")
        return {"messages": [], "note": "success on retry"}

    agent._ask_impl = fake_ask_impl
    agent._key_index = 0  # ensure there IS a next key available to rotate to

    result = agent.ask("test question", thread_id="test-rotation")
    assert call_count["n"] == 2, "should have retried exactly once after rotating"
    assert result["note"] == "success on retry"
    print(f"PASS — ask() caught the exhaustion, rotated (now on key index "
          f"{agent._key_index}), and retried successfully\n")


def test_no_keys_left_reraises():
    print("=== Test 4: no keys left to rotate to -> re-raises instead of retrying forever ===")

    def always_fails(question, thread_id="default"):
        raise agent.DailyQuotaExhausted("simulated quota exhaustion")

    agent._ask_impl = always_fails
    agent._key_index = len(agent.KEY_POOL) - 1  # already on the last configured key

    try:
        agent.ask("test question", thread_id="test-rotation-2")
        print("FAIL — should have raised DailyQuotaExhausted\n")
    except agent.DailyQuotaExhausted:
        print("PASS — correctly re-raised when no keys remain to rotate to\n")


if __name__ == "__main__":
    test_persistence_across_simulated_runs()
    test_stale_index_discarded()
    test_reactive_rotation()
    test_no_keys_left_reraises()

    # Clean up — don't let test state leak into tomorrow's real rotation
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    print("Cleaned up test state file — real rotation starts fresh from key 0 "
          "on your next actual run, same as before this test.")