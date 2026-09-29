"""Replay-diet Stage-1 code levers (CONTEXT-DIET-SPEC-v1 §8 Stage 1, card t_4311f16c).

Lever (a): tool_calls echo trim — oversized historical assistant tool-call argument
echoes trimmed on the WIRE COPY only (persisted rows untouched, JSON stays valid,
deterministic/idempotent so the prompt-cache prefix stays stable after one boundary).

Lever (b): distill-beyond-window stubs — the proactive prune demotes out-of-window
tool results to one-line stubs carrying recall pointers (result_id, tool, gist, byte
location); full bytes survive in state.db via the archive commit.
"""

from __future__ import annotations

import copy
import json

import pytest

from agent.context_compressor import (
    _DISTILL_STUB_PREFIX,
    _is_summary_stub,
    _summarize_tool_result,
    ContextCompressor,
)
from agent.replay_diet import (
    DEFAULT_TOOL_CALLS_MAX_ARGS_CHARS,
    get_replay_diet_limits,
    trim_tool_calls_echoes,
)

LARGE_WINDOW = 1_000_000


def _compressor(**kw):
    from unittest.mock import patch

    defaults = dict(
        model="test", quiet_mode=True, threshold_percent=0.50,
        protect_first_n=2, protect_last_n=4,
        proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000,
    )
    defaults.update(kw)
    with patch("agent.context_compressor.get_model_context_length", return_value=LARGE_WINDOW):
        return ContextCompressor(**defaults)


# ---------------------------------------------------------------------------
# Lever (a) — tool_calls echo trim (pure functions)
# ---------------------------------------------------------------------------

def _big_args(n_chars: int = 20_000) -> str:
    return json.dumps({"code": "x" * n_chars, "label": "probe"})


def _assistant_call(cid, args):
    return {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": cid, "type": "function",
                        "function": {"name": "execute_code", "arguments": args}}],
    }


def test_trim_returns_none_shape_when_disabled():
    msgs = [_assistant_call("c1", _big_args())]
    snapshot = copy.deepcopy(msgs)
    assert trim_tool_calls_echoes(msgs, 0) == 0
    assert trim_tool_calls_echoes([], 200) == 0
    assert msgs == snapshot  # input never mutated when off


def test_trim_shrinks_oversized_echo_and_keeps_json_valid():
    msgs = [_assistant_call("c1", _big_args())]
    trimmed = trim_tool_calls_echoes(msgs, 200)
    assert trimmed == 1
    args = msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert len(args) < 1_000  # 20k chars collapsed to head + marker
    parsed = json.loads(args)  # MUST stay valid JSON — providers 400 otherwise
    assert "code" in parsed


def test_trim_is_deterministic_and_idempotent():
    one = [_assistant_call("c1", _big_args(9_999))]
    two = [_assistant_call("c1", _big_args(9_999))]
    assert trim_tool_calls_echoes(one, 200) == 1
    assert trim_tool_calls_echoes(two, 200) == 1
    assert one[0]["tool_calls"][0]["function"]["arguments"] == \
        two[0]["tool_calls"][0]["function"]["arguments"]
    # Second pass over already-trimmed rows: no further change (marker recognized).
    args_after_first = one[0]["tool_calls"][0]["function"]["arguments"]
    assert trim_tool_calls_echoes(one, 200) == 0
    assert one[0]["tool_calls"][0]["function"]["arguments"] == args_after_first


def test_trim_small_args_untouched():
    small = json.dumps({"cmd": "ls"})
    msgs = [_assistant_call("c1", small)]
    assert trim_tool_calls_echoes(msgs, 200) == 0
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == small


def test_trim_respects_history_end_idx_live_rows_spared():
    msgs = [_assistant_call("c1", _big_args()), _assistant_call("c2", _big_args())]
    # Only the first (historical) row is eligible; the live row (index 1) is spared.
    assert trim_tool_calls_echoes(msgs, 200, history_end_idx=1) == 1
    assert len(json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"])["code"]) == 20_000


def test_trim_malformed_args_left_alone():
    msgs = [_assistant_call("c1", "{not json at all " + "x" * 5_000)]
    assert trim_tool_calls_echoes(msgs, 200) == 0


def test_default_config_is_off():
    assert DEFAULT_TOOL_CALLS_MAX_ARGS_CHARS == 0


# ---------------------------------------------------------------------------
# Lever (a) — build_api_messages integration (wire copy, persisted rows untouched)
# ---------------------------------------------------------------------------

class _DietAgent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 10_000.0

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _build_request(agent, history, current_turn_user_idx):
    from agent.turn_context import build_api_messages

    request, _ = build_api_messages(
        agent, history, current_turn_user_idx=current_turn_user_idx,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None,
        active_system_prompt="",
    )
    return request


def test_build_api_messages_trims_only_replayed_prefix(tmp_path, monkeypatch):
    """Lever (a) on the send path: historical echoes trimmed on the wire copy; the
    current turn's live rows and the persisted history stay byte-exact."""
    import agent.replay_diet as rd

    monkeypatch.setattr(rd, "get_replay_diet_limits", lambda: 200)
    big = _big_args(15_000)
    history = [
        {"role": "user", "content": "run this"},
        _assistant_call("c1", big),
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "user", "content": "go"},                    # <- current turn's user message
        _assistant_call("c2", big),                          # live rows appended THIS turn
        {"role": "tool", "tool_call_id": "c2", "content": "ok2"},
    ]
    snapshot = copy.deepcopy(history)
    request = _build_request(_DietAgent(), history, current_turn_user_idx=3)
    # Historical echo (c1, before the current user message) trimmed on the wire copy...
    hist_args = request[1]["tool_calls"][0]["function"]["arguments"]
    assert len(hist_args) < 1_000
    json.loads(hist_args)  # still valid
    # ...but the CURRENT turn's live call (c2, appended after the user message) is NOT
    # trimmed — it stays the full 15k-char payload (modulo the pre-existing wire JSON
    # canonicalization: sort_keys + compact separators, untouched by this feature).
    live_msgs = [m for m in request if m.get("role") == "assistant" and m.get("tool_calls")]
    live_args = json.loads(live_msgs[-1]["tool_calls"][0]["function"]["arguments"])
    assert len(live_args["code"]) == 15_000
    # Persisted history untouched.
    assert history == snapshot


def test_build_api_messages_noop_when_unset(tmp_path, monkeypatch):
    import agent.replay_diet as rd

    monkeypatch.setattr(rd, "get_replay_diet_limits", lambda: 0)
    big = _big_args(15_000)
    history = [
        {"role": "user", "content": "go"},
        _assistant_call("c1", big),
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    request = _build_request(_DietAgent(), history, current_turn_user_idx=0)
    assert request[1]["tool_calls"][0]["function"]["arguments"] == big


# ---------------------------------------------------------------------------
# Lever (b) — distill-beyond-window stubs with recall pointers
# ---------------------------------------------------------------------------

def _prune_history(n_pairs=8, big_indices=(0, 1, 2), big_chars=9_000):
    msgs = [{"role": "user", "content": "start"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid, json.dumps({"command": f"step-{i}"})))
        content = chr(65 + (i % 26)) * big_chars if i in big_indices else "ok"
        msgs.append({"role": "role-placeholder", "tool_call_id": cid, "content": content} if False else
                    {"role": "tool", "tool_call_id": cid, "content": content})
    return msgs


def test_proactive_prune_emits_recall_pointer_stubs():
    c = _compressor()
    msgs = _prune_history()
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned >= 3
    stub = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "call_0"][0]
    content = stub["content"]
    # Stub shape: sigil + the three recall pointers + byte location.
    assert content.startswith(_DISTILL_STUB_PREFIX)
    assert "result_id=call_0" in content
    assert "tool=execute_code" in content  # tool name pointer
    assert "state.db" in content           # byte-location pointer
    assert len(content) < 400              # a true one-liner
    # gist carried: the deterministic summarizer line survives inside the stub.
    assert "chars" in content


def test_distill_stub_is_idempotent_under_repeated_prunes():
    c = _compressor()
    msgs = _prune_history()
    first, n1 = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert n1 >= 3
    second, n2 = c.prune_tool_results_only(first, current_tokens=None)
    assert n2 == 0
    assert [m.get("content") for m in second] == [m.get("content") for m in first]


def test_is_summary_stub_recognizes_distill_stub():
    stub = _summarize_tool_result("terminal", "{}", "x" * 9_000)
    assert _is_summary_stub("[pruned: result_id=call_1 tool=terminal terminal ran `ls` -> exit 0, 9 lines output — full 9,000 chars in state.db (session_search result_id:call_1 or tool-role LIKE recall)]")
    assert not _is_summary_stub("regular tool output body that is long enough to not look like a stub at all..." * 5)


def test_distill_keeps_call_result_pairing():
    c = _compressor()
    msgs = _prune_history(n_pairs=10, big_indices=(0, 1, 2, 3, 4))
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned >= 1
    call_ids = set()
    for m in result:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                call_ids.add(tc["id"] if isinstance(tc, dict) else tc.id)
    result_ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
    assert result_ids <= call_ids, "orphan tool results without a matching call"
    assert call_ids <= result_ids, "orphan tool calls without a matching result"


def test_full_compression_prune_keeps_plain_summary_shape():
    """Only the proactive (beyond-window distill) path emits pointer stubs; the
    full-compression pass keeps its established plain-summary output shape."""
    c = _compressor()
    msgs = _prune_history()
    result, pruned = c._prune_old_tool_results(msgs, protect_tail_count=4)
    assert pruned >= 1
    demoted = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "call_0"][0]
    assert not demoted["content"].startswith(_DISTILL_STUB_PREFIX)


def test_distill_stub_deterministic():
    from agent.context_compressor import _distill_result_stub

    a = _distill_result_stub("call_9", "terminal", '{"cmd":"ls"}', "z" * 9_000)
    b = _distill_result_stub("call_9", "terminal", '{"cmd":"ls"}', "z" * 9_000)
    assert a == b and a.startswith(_DISTILL_STUB_PREFIX)
    # Result id and tool ride the stub.
    assert "result_id=call_9" in a and "tool=terminal" in a