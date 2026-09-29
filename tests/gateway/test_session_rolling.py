"""ROLLING SESSIONS — threshold-triggered rotation with handoff (agent.session_rolling).

Justin's directive (2026-09-29, thread 45962): sessions close and reopen; the user sees one
timeline, the machine runs multiple sessions under it; on reopen, context arrives via a
HANDOFF (compression summary + verbatim tail), not a full-history re-read.

The compression engine's rotation mode (``compression_in_place=False``) already implements
the handoff: parent row preserved in state.db (searchable, /resume-able), child session
opening with summary + verbatim tail. Gateway hygiene hard-forced IN-PLACE compaction
(``compression_in_place=True``), which keeps the SAME session alive: 407→108 msgs = ~83k
tokens, still growing back — a brake, not a fix. The full history keeps being re-read on
every subsequent turn until the next compaction.

These tests bind the switch:

1. RED-side (defect proof): with rolling OFF (the default), hygiene forces in-place
   compaction — the session never rotates, the transcript regrows.
2. GREEN: with ``agent.session_rolling.enabled: true`` hygiene runs the engine's rotation
   mode; the live entry, the held turn lease, and the Telegram topic lane rebind to the
   child; the child transcript IS the handoff.
3. Config parsing: defaults off, overrides honored, invalid values fail-soft.
"""

import asyncio
import importlib
import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_history(n_messages: int, content_size: int = 100) -> list:
    history = []
    content = "x" * content_size
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": content, "timestamp": f"t{i}"})
    return history


class HygieneCaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="rolling-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _install_fake_dotenv(monkeypatch):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)


def _rolling_config(tmp_path, monkeypatch, gateway_run, rolling_cfg=None, extra=None):
    """Write a config.yaml with the given agent.session_rolling block; _hermes_home → tmp_path."""
    import yaml

    cfg = {"model": {"default": "anthropic/claude-sonnet-4.6"}}
    if rolling_cfg is not None:
        cfg["agent"] = {"session_rolling": rolling_cfg}
    if extra:
        cfg.update(extra)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)


def _make_runner(tmp_path, monkeypatch, gateway_run, adapter, session_id="sess-1",
                 chat_type="private", thread_id=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type=chat_type,
    )
    runner.session_store.load_transcript.return_value = _make_history(12, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )
    return runner


def _telegram_event(chat_type="private", thread_id=None):
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type=chat_type,
            thread_id=thread_id,
            user_id="12345",
        ),
        message_id="1",
    )


# ---------------------------------------------------------------------------
# RED: the defect — default (rolling OFF) keeps the session alive in-place
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rolling_off_default_forces_in_place_compaction(monkeypatch, tmp_path):
    """RED-side proof of the pre-change burn: with agent.session_rolling unset (default OFF,
    exactly the pre-change behavior), hygiene forces compression_in_place=True. The SAME
    session_id survives; the compacted transcript stays inside one ever-regrowing session
    that is re-read on every turn — the full-history-per-turn burn the card measures
    (407→108 msgs = ~83k tokens, still growing back)."""
    _install_fake_dotenv(monkeypatch)
    gateway_run = importlib.import_module("gateway.run")

    seen_modes = {}

    class FakeInPlaceAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.platform = kwargs.get("platform")
            self.session_id = kwargs.get("session_id", "sess-1")
            self._session_db = kwargs.get("session_db")
            self.compression_in_place = False
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
                protect_last_n=20,
            )
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            seen_modes["in_place"] = self.compression_in_place
            self._last_compaction_in_place = True
            return ([{"role": "assistant", "content": "compacted"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeInPlaceAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None
    adapter = HygieneCaptureAdapter()
    runner = _make_runner(tmp_path, monkeypatch, gateway_run, adapter)
    runner._session_db = SimpleNamespace(_db=fake_db, get_session=AsyncMock(return_value={}))

    _rolling_config(tmp_path, monkeypatch, gateway_run, rolling_cfg=None)

    result = await runner._handle_message(_telegram_event())

    assert result == "ok"
    # The defect, on the record: rolling OFF ⇒ in-place ⇒ the same session lives on.
    assert seen_modes["in_place"] is True
    assert FakeInPlaceAgent.last_instance.session_id == "sess-1"
    runner.session_store.rewrite_transcript.assert_not_called()


# ---------------------------------------------------------------------------
# GREEN: rolling ON rotates the session and the child transcript IS the handoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rolling_enabled_rotates_session_with_handoff(monkeypatch, tmp_path):
    """GREEN: agent.session_rolling.enabled=true routes hygiene through the engine's
    rotation mode. The live entry rebinds to the child session (new session_id under the
    same session_key), and the child transcript written for the next turn is the
    HANDOFF (summary + verbatim tail) — per-turn input is handoff + window, never the
    regrown full history."""
    _install_fake_dotenv(monkeypatch)
    gateway_run = importlib.import_module("gateway.run")

    handoff_transcript = [
        {"role": "user", "content": "[Conversation continued from a previous session]\n"
                                   "HANDOFF SUMMARY: user was testing rolling sessions."},
        {"role": "assistant", "content": "Understood, continuing."},
    ] + _make_history(20, content_size=50)  # the verbatim tail

    class FakeRotatingAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.platform = kwargs.get("platform")
            self.session_id = kwargs.get("session_id", "sess-1")
            self._session_db = kwargs.get("session_db")
            self.compression_in_place = True
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
                protect_last_n=20,
            )
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            assert self.compression_in_place is False, (
                "rolling sessions must run the engine's rotation mode, not in-place"
            )
            # Engine rotation contract: the agent is re-pointed to the minted child id.
            self.session_id = "child-rotated-1"
            self._last_compaction_in_place = False
            return (list(handoff_transcript), None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeRotatingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None
    adapter = HygieneCaptureAdapter()
    runner = _make_runner(tmp_path, monkeypatch, gateway_run, adapter)
    runner._session_db = SimpleNamespace(_db=fake_db, get_session=AsyncMock(return_value={}))

    _rolling_config(tmp_path, monkeypatch, gateway_run,
                    rolling_cfg={"enabled": True})

    result = await runner._handle_message(_telegram_event())

    assert result == "ok"
    entry = runner.session_store.get_or_create_session.return_value
    # New session_id under the SAME session_key: the user's timeline never changes.
    assert entry.session_id == "child-rotated-1"
    assert entry.session_key == "agent:main:telegram:private:12345"
    # The child transcript IS the handoff: summary row + verbatim tail, NOT the old history.
    runner.session_store.rewrite_transcript.assert_called_once()
    written_sid, written_transcript = runner.session_store.rewrite_transcript.call_args[0]
    assert written_sid == "child-rotated-1"
    assert len(written_transcript) == len(handoff_transcript)
    assert "HANDOFF SUMMARY" in written_transcript[0]["content"]
    # Token burn measured at handoff size, not full history size.
    assert (estimate_messages_tokens_rough(written_transcript)
            < estimate_messages_tokens_rough(runner.session_store.load_transcript.return_value))


@pytest.mark.asyncio
async def test_rolling_rotation_failure_fails_closed_to_original(monkeypatch, tmp_path):
    """Fail-closed: a rotation whose child transcript cannot be persisted keeps the live
    entry on the ORIGINAL session — the conversation is never dropped (#21301 guard,
    already enforced by the production adopt path; this test binds it for rolling)."""
    _install_fake_dotenv(monkeypatch)
    gateway_run = importlib.import_module("gateway.run")

    class FakeRotatingAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "sess-1")
            self.compression_in_place = True
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
                protect_last_n=20,
            )
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            self.session_id = "child-rotated-1"
            return ([{"role": "user", "content": "HANDOFF SUMMARY: broken persist"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeRotatingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None
    adapter = HygieneCaptureAdapter()
    runner = _make_runner(tmp_path, monkeypatch, gateway_run, adapter)
    runner._session_db = SimpleNamespace(_db=fake_db, get_session=AsyncMock(return_value={}))
    # Persist failure: the child transcript never lands.
    runner.session_store.rewrite_transcript.return_value = False

    _rolling_config(tmp_path, monkeypatch, gateway_run, rolling_cfg={"enabled": True})

    result = await runner._handle_message(_telegram_event())

    assert result == "ok"
    # Fail-closed: the entry stays on the original session; nothing was dropped.
    entry = runner.session_store.get_or_create_session.return_value
    assert entry.session_id == "sess-1"


# ---------------------------------------------------------------------------
# Config parsing: defaults, overrides, fail-soft
# ---------------------------------------------------------------------------

class TestRollingSessionConfig:
    def _hs(self, data):
        from gateway.run_turn import GatewayTurnMixin

        mixin = GatewayTurnMixin.__new__(GatewayTurnMixin)
        hs = SimpleNamespace(
            model="anthropic/claude-sonnet-4.6",
            provider=None,
            base_url=None,
            config_context_length=None,
            compression_enabled=True,
            hard_msg_limit=5000,
            timeout_seconds=30.0,
            total_ceiling_seconds=600.0,
            max_turn_hold_seconds=10.0,
            failure_cooldown_seconds=300.0,
            threshold_pct=0.85,
            session_rolling_enabled=False,
            session_rolling_threshold=None,
            session_rolling_tail_n=None,
            session_rolling_keep_verbatim=True,
        )
        mixin._hmwa_hygiene_read_config(hs, data)
        return hs

    def test_default_off(self):
        hs = self._hs({})
        assert hs.session_rolling_enabled is False
        assert hs.session_rolling_threshold is None
        assert hs.session_rolling_tail_n is None
        assert hs.session_rolling_keep_verbatim is True

    def test_enabled_true(self):
        hs = self._hs({"agent": {"session_rolling": {"enabled": True}}})
        assert hs.session_rolling_enabled is True

    def test_threshold_fraction_and_percent(self):
        hs = self._hs({"agent": {"session_rolling": {"enabled": True, "threshold": 0.5}}})
        assert hs.session_rolling_threshold == 0.5
        hs = self._hs({"agent": {"session_rolling": {"enabled": True, "threshold": 70}}})
        assert hs.session_rolling_threshold == 0.7

    def test_invalid_threshold_fail_soft(self):
        hs = self._hs({"agent": {"session_rolling": {"enabled": True, "threshold": 0}}})
        assert hs.session_rolling_threshold is None
        hs = self._hs({"agent": {"session_rolling": {"enabled": True, "threshold": "junk"}}})
        assert hs.session_rolling_threshold is None

    def test_tail_n_and_keep_verbatim(self):
        hs = self._hs({"agent": {"session_rolling": {
            "enabled": True, "tail_n": 30, "keep_verbatim": False,
        }}})
        assert hs.session_rolling_tail_n == 30
        assert hs.session_rolling_keep_verbatim is False

    def test_bad_types_fail_soft(self):
        hs = self._hs({"agent": {"session_rolling": {
            "enabled": True, "tail_n": "many", "keep_verbatim": "no",
        }}})
        assert hs.session_rolling_tail_n is None
        assert hs.session_rolling_keep_verbatim is False
        hs = self._hs({"agent": "not-a-dict"})
        assert hs.session_rolling_enabled is False


@pytest.mark.asyncio
async def test_rolling_tail_n_override_reaches_compressor(monkeypatch, tmp_path):
    """tail_n / keep_verbatim are consumer-side knobs: the detached hygiene compressor's
    protect_last_n is reconfigured before the attempt (engine behavior unchanged)."""
    _install_fake_dotenv(monkeypatch)
    gateway_run = importlib.import_module("gateway.run")

    class FakeTailAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "sess-1")
            self.compression_in_place = True
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
                protect_last_n=20,
            )
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            return ([{"role": "user", "content": "HANDOFF SUMMARY"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeTailAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None
    adapter = HygieneCaptureAdapter()
    runner = _make_runner(tmp_path, monkeypatch, gateway_run, adapter)
    runner._session_db = SimpleNamespace(_db=fake_db, get_session=AsyncMock(return_value={}))

    _rolling_config(tmp_path, monkeypatch, gateway_run,
                    rolling_cfg={"enabled": True, "tail_n": 30, "keep_verbatim": False})

    await runner._handle_message(_telegram_event())

    agent = FakeTailAgent.last_instance
    # keep_verbatim=false wins (summary-first); the engine still enforces its own floor.
    assert agent.context_compressor.protect_last_n == 0


@pytest.mark.asyncio
async def test_rolling_threshold_override_changes_trigger_point(monkeypatch, tmp_path):
    """agent.session_rolling.threshold overrides the rotation trigger point."""
    _install_fake_dotenv(monkeypatch)
    gateway_run = importlib.import_module("gateway.run")

    class NeverCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "sess-1")
            self.compression_in_place = True
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
                protect_last_n=20,
            )
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            raise AssertionError("must not compress: threshold 0.99 should not fire at ~1k tokens")

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = NeverCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None
    adapter = HygieneCaptureAdapter()
    runner = _make_runner(tmp_path, monkeypatch, gateway_run, adapter)
    runner._session_db = SimpleNamespace(_db=fake_db, get_session=AsyncMock(return_value={}) )

    # threshold 0.99 of a 100-token context = 99 tokens; the ~1.2k-token history WOULD fire
    # at the default 0.85 — this asserts the override actually moved the trigger.
    _rolling_config(tmp_path, monkeypatch, gateway_run,
                    rolling_cfg={"enabled": True, "threshold": 0.99},
                    extra={"model": {"default": "anthropic/claude-sonnet-4.6",
                                     "context_length": 100}})

    # Force the plan's context length to 100 via the metadata monkeypatch already applied
    # in _make_runner; history ~1.2k tokens ≥ 85 (default would fire) but < 99.
    result = await runner._handle_message(_telegram_event())
    assert result == "ok"
    runner.session_store.rewrite_transcript.assert_not_called()
    assert NeverCompressAgent.last_instance.session_id == "sess-1"