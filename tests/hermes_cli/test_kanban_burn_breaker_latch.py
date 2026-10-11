"""Burn-breaker spawn latch in the kanban dispatcher (t_8973af47).

An external cost circuit breaker latches by writing ``{"tripped": true}`` to
its JSON state file. Before this guard the dispatcher kept claiming ready
cards and spawning workers that the breaker SIGTERMed minutes later — a
3-day spawn/kill churn. The latch makes ``_tick_spawn_budget`` refuse new
spawns while the file says tripped.

Contract under test:

* ``tripped: true``  -> 0 spawns, ``result.burn_breaker_latched``, a WARNING
  naming the latch; cards stay ``ready`` (deferred, not dropped).
* ``tripped: false`` -> normal spawns.
* missing / unreadable / malformed / non-object file -> FAIL OPEN: normal
  spawns plus a WARNING (a broken breaker file must never freeze the roster).
* path: ``HERMES_BURN_BREAKER_STATE`` env > ``kanban.burn_breaker_state_file``
  config > built-in default; ``off``/``none``/``disabled`` disables the check.
* cheap: the file is re-parsed only when its stat signature changes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

pytestmark = pytest.mark.real_burn_breaker

ENV = "HERMES_BURN_BREAKER_STATE"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    # Each test starts with a cold latch cache and a fresh warn-once set.
    reset = getattr(kbd, "_reset_burn_breaker_cache", None)
    if reset is not None:
        reset()
    return home


def _write_state(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def _dispatch_three(spawns):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        ids = [kb.create_task(conn, title=t, assignee="alice") for t in ("a", "b", "c")]
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        statuses = {tid: kb.get_task(conn, tid).status for tid in ids}
    return res, ids, statuses


def test_tripped_true_spawns_nothing_and_logs(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path, caplog,
):
    state = _write_state(tmp_path / "breaker.json", {"tripped": True, "trip_ts": "x"})
    monkeypatch.setenv(ENV, str(state))
    spawns: list[str] = []
    with caplog.at_level(logging.WARNING):
        res, ids, statuses = _dispatch_three(spawns)

    assert spawns == []
    assert res.spawned == []
    assert getattr(res, "burn_breaker_latched", False) is True
    # Deferred, not dropped: the cards are still claimable.
    assert set(statuses.values()) == {"ready"}
    assert any(
        "burn breaker latched" in r.getMessage() and str(state) in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert "burn_breaker_latched" in kbd.describe_suppression([res])


def test_tripped_false_spawns_normally(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path, caplog,
):
    state = _write_state(tmp_path / "breaker.json", {"tripped": False})
    monkeypatch.setenv(ENV, str(state))
    spawns: list[str] = []
    with caplog.at_level(logging.WARNING):
        res, ids, _ = _dispatch_three(spawns)

    assert sorted(spawns) == sorted(ids)
    assert not getattr(res, "burn_breaker_latched", False)
    assert not any("burn breaker" in r.getMessage() for r in caplog.records)


def test_rearm_resumes_spawning_on_next_tick(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path,
):
    state = _write_state(tmp_path / "breaker.json", {"tripped": True})
    monkeypatch.setenv(ENV, str(state))
    spawns: list[str] = []
    res, ids, _ = _dispatch_three(spawns)
    assert spawns == []

    # Breaker --rearm rewrites the file (atomic rename in practice).
    tmp = state.with_suffix(".tmp")
    tmp.write_text(json.dumps({"tripped": False, "sustain_over": 0}), encoding="utf-8")
    os.replace(tmp, state)

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
    assert sorted(spawns) == sorted(ids)
    assert not res2.burn_breaker_latched


@pytest.mark.parametrize(
    "label,content",
    [
        ("missing", None),
        ("malformed_json", "{not json"),
        ("non_object", "[true]"),
        ("empty", ""),
    ],
)
def test_broken_state_file_fails_open_with_warning(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path, caplog, label, content,
):
    state = tmp_path / "breaker.json"
    if content is not None:
        _write_state(state, content)
    monkeypatch.setenv(ENV, str(state))
    spawns: list[str] = []
    with caplog.at_level(logging.WARNING):
        res, ids, _ = _dispatch_three(spawns)

    assert sorted(spawns) == sorted(ids), label
    assert not getattr(res, "burn_breaker_latched", False)
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "burn breaker" in r.getMessage()
        and "fail" in r.getMessage().lower()
    ]
    assert warnings, (label, [r.getMessage() for r in caplog.records])


def test_unreadable_state_file_fails_open(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path,
):
    # A directory at the path: stat succeeds, read raises IsADirectoryError /
    # PermissionError depending on platform — either way an OSError.
    state = tmp_path / "breaker.json"
    state.mkdir()
    monkeypatch.setenv(ENV, str(state))
    spawns: list[str] = []
    res, ids, _ = _dispatch_three(spawns)
    assert sorted(spawns) == sorted(ids)
    assert not res.burn_breaker_latched


def test_broken_file_warning_is_not_repeated_every_tick(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path, caplog,
):
    monkeypatch.setenv(ENV, str(tmp_path / "absent.json"))
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert kbd.burn_breaker_latched() is False
    hits = [r for r in caplog.records if "burn breaker" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in hits]


def test_disabled_via_env(kanban_home, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(ENV, "off")
    with caplog.at_level(logging.WARNING):
        assert kbd.burn_breaker_latched() is False
    assert not caplog.records


def test_config_key_used_when_env_unset(kanban_home, monkeypatch, tmp_path):
    state = _write_state(tmp_path / "cfg-breaker.json", {"tripped": True})
    monkeypatch.delenv(ENV, raising=False)
    from hermes_cli import config as cfg

    monkeypatch.setattr(
        cfg, "load_config_readonly",
        lambda: {"kanban": {"burn_breaker_state_file": str(state)}},
    )
    assert kbd.burn_breaker_state_path() == state
    assert kbd.burn_breaker_latched() is True


def test_env_beats_config(kanban_home, monkeypatch, tmp_path):
    env_state = _write_state(tmp_path / "env.json", {"tripped": False})
    cfg_state = _write_state(tmp_path / "cfg.json", {"tripped": True})
    monkeypatch.setenv(ENV, str(env_state))
    from hermes_cli import config as cfg

    monkeypatch.setattr(
        cfg, "load_config_readonly",
        lambda: {"kanban": {"burn_breaker_state_file": str(cfg_state)}},
    )
    assert kbd.burn_breaker_state_path() == env_state
    assert kbd.burn_breaker_latched() is False


def test_default_path_is_home_relative(kanban_home, monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    from hermes_cli import config as cfg

    monkeypatch.setattr(cfg, "load_config_readonly", lambda: {})
    assert kbd.burn_breaker_state_path() == (
        tmp_path / "viiy-hq" / "personal" / "context-sync" / "costs"
        / "burn-breaker-state.json"
    )


def test_unchanged_file_is_not_reparsed(kanban_home, monkeypatch, tmp_path):
    state = _write_state(tmp_path / "breaker.json", {"tripped": True})
    monkeypatch.setenv(ENV, str(state))
    calls = {"n": 0}
    real = kbd._parse_burn_breaker_state

    def counting(raw):
        calls["n"] += 1
        return real(raw)

    monkeypatch.setattr(kbd, "_parse_burn_breaker_state", counting)
    for _ in range(5):
        assert kbd.burn_breaker_latched() is True
    assert calls["n"] == 1

    _write_state(state, {"tripped": False, "pad": "changes size"})
    assert kbd.burn_breaker_latched() is False
    assert calls["n"] == 2
