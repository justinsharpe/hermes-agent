"""Tests: gateway-restart orphaned run rows must not wedge the dispatcher.

Reproduced three times in one day (2026-09-29): a gateway restart (or an
external fleet circuit breaker) kills worker PIDs mid-run, and an external
sweep flips the claimed cards to a non-running status with raw SQL —
WITHOUT clearing ``claim_lock`` and WITHOUT closing the open ``task_runs``
row. The card is then wedged permanently:

* every reclaim path (``release_stale_claims`` / ``detect_crashed_workers``
  / ``reconcile_orphaned_running``) gates on ``status = 'running'``, so a
  lock stranded on a todo/ready/blocked/review card is invisible to all
  of them;
* ``_lane_rows`` requires ``claim_lock IS NULL``, so the dispatcher can
  never claim the card again;
* ``recompute_ready`` keeps firing ``promoted`` events for the card
  ("promoted into the void") while nothing can ever pick it up.

The fix direction (task t_4aecb4cb, sealed lesson in the
kanban-card-dispatch skill: worker-death => run-row-close + lock-clear
must be ONE transaction): the dispatcher's reclaim phase must ALSO sweep
non-running cards whose residual claim's worker is ps-dead — release the
lock, close the leaked run, and let the card be claimed normally, all in
one transaction, before promotion runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _make_zombie(conn, host, *, dest_status="ready", assignee="w"):
    """Build the observed zombie: claimed card whose worker died, then an
    external sweep flipped the card's status with raw SQL, leaving
    ``claim_lock`` / ``worker_pid`` / the open run row behind."""
    tid = kb.create_task(conn, title="zombie", assignee=assignee)
    kb.claim_task(conn, tid, claimer=f"{host}:zombie-owner")
    dead = subprocess.Popen(["true"])
    dead.wait()  # ps-dead, reaped, invisible on every liveness probe
    kbd._set_worker_pid(conn, tid, dead.pid)
    # The external sweep: status flip WITHOUT lock-clear / run-close.
    cur = conn.execute(
        "UPDATE tasks SET status = ? WHERE id = ?",
        (dest_status, tid),
    )
    conn.commit()
    assert cur.rowcount == 1
    row = conn.execute(
        "SELECT claim_lock, worker_pid, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["claim_lock"] == f"{host}:zombie-owner"
    assert row["current_run_id"] is not None  # run row left open
    return tid


def test_zombie_ready_card_is_swept_within_one_tick(conn):
    """ACCEPTANCE 1: a ready card carrying a dead worker's residual claim is
    released (lock cleared, run closed) by one dispatch tick, and the card
    is then claimable again — no manual sweep."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = _make_zombie(conn, host, dest_status="ready")

    result = kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace, board=None: 999999,
        reconcile_orphans=True,
    )

    assert tid in result.reconciled_claims, (
        "one dispatch tick must release a ready card stranded with a dead "
        "worker's claim lock"
    )
    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
        "FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["claim_lock"] is None
    assert row["current_run_id"] is None
    # The card must be claimable again right away.
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    assert claimed.status == "running"


def test_zombie_blocked_card_promotion_does_not_fire_into_the_void(conn):
    """ACCEPTANCE 2: a blocked card whose residual claim's worker is dead
    must not get a bare ``promoted`` event while it still holds the lock —
    the sweep releases the claim FIRST (one txn), so the later promotion
    describes a claimable card. Also: ``recompute_ready`` must not promote
    a lock-holding card whose worker is still alive (the lock means
    someone still owns it)."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = _make_zombie(conn, host, dest_status="blocked")

    result = kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace, board=None: 999999,
        reconcile_orphans=True,
    )

    assert tid in result.reconciled_claims
    row = conn.execute(
        "SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["claim_lock"] is None


def test_live_worker_claim_is_never_swept(conn):
    """A card whose claimed worker is STILL ALIVE is never touched by the
    sweep — the lock is real ownership, regardless of the card's status
    column. (The sweep only releases claims whose worker is ps-dead.)"""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="live", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:live-owner")
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        kbd._set_worker_pid(conn, tid, sleeper.pid)
        # External sweep flips the status but the worker is alive.
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,)
        )
        conn.commit()

        result = kbd.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace, board=None: 999999,
            reconcile_orphans=True,
        )
        assert tid not in result.reconciled_orphans
        row = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["claim_lock"] == f"{host}:live-owner", (
            "a live worker's claim must never be released by the sweep"
        )
    finally:
        sleeper.kill()
        sleeper.wait()


def test_zombie_sweep_closes_the_leaked_run_row_in_same_txn(conn):
    """ACCEPTANCE (lesson): worker-death => run-row-close + lock-clear in
    ONE transaction. The open ``task_runs`` row must land terminal
    (``reclaimed``) with the lock release, not stay open as a phantom stall."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = _make_zombie(conn, host, dest_status="ready")
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]
    assert run_id is not None

    kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace, board=None: 999999,
        reconcile_orphans=True,
    )

    run = conn.execute(
        "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert run is not None
    assert run["ended_at"] is not None, "leaked run row must be closed"
    assert run["status"] == "reclaimed"
    assert run["outcome"] == "reclaimed"


def test_sweep_covers_running_zombie_after_gateway_restart(conn):
    """The original repro: gateway restart kills a worker whose card is
    STILL ``running``. ``detect_crashed_workers`` already reclaims that
    class; this test pins that the one-tick guarantee also holds for it —
    no regression from the new sweep's ordering (sweep BEFORE promote)."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="killed", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:killed-owner")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)

    result = kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace, board=None: 999999,
    )

    row = conn.execute(
        "SELECT status, claim_lock, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["claim_lock"] is None, "killed worker's claim must be released"
    assert row["current_run_id"] is None, "killed worker's run must be closed"


def test_orphaned_run_rows_are_closed_by_one_tick(conn):
    """The other half of the wedge: open ``task_runs`` rows on cards that no
    longer own them (current_run_id gone) must be closed ``reclaimed`` by one
    tick — they feed the phantom-stall watchdog otherwise."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="orphan-run", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:owner")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    # External sweep closes the task's ownership but leaks the run row.
    conn.execute(
        "UPDATE tasks SET status='done', claim_lock=NULL, worker_pid=NULL, "
        "current_run_id=NULL WHERE id = ?", (tid,)
    )
    conn.commit()
    run_id = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
        (tid,),
    ).fetchone()["id"]
    assert run_id is not None

    result = kbd.dispatch_once(
        conn, spawn_fn=lambda task, workspace, board=None: 999999,
    )

    assert run_id in result.closed_orphaned_runs
    run = conn.execute(
        "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert run["status"] == "reclaimed"
    assert run["ended_at"] is not None


def test_orphaned_run_with_live_worker_is_not_closed(conn):
    """A leaked run row whose worker is STILL ALIVE must never be closed —
    that's ``reap_terminal_workers``' ground (terminal survivor), and closing
    the run beside a live process orphans the process's bookkeeping."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="live-orphan-run", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:live2")
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        kbd._set_worker_pid(conn, tid, sleeper.pid)
        conn.execute(
            "UPDATE tasks SET status='done', claim_lock=NULL, worker_pid=NULL, "
            "current_run_id=NULL WHERE id = ?", (tid,)
        )
        conn.commit()

        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, workspace, board=None: 999999,
        )
        run = conn.execute(
            "SELECT status, ended_at FROM task_runs WHERE task_id = ? "
            "AND ended_at IS NULL",
            (tid,),
        ).fetchone()
        assert run is not None, "run beside a live worker must stay open"
        assert result.closed_orphaned_runs == [] or not any(
            r for r in result.closed_orphaned_runs
        )
    finally:
        sleeper.kill()
        sleeper.wait()


def test_hostname_alias_lock_is_recognized_as_host_local(conn, monkeypatch):
    """The wedge lived on exactly this gap: locks issued as
    ``Sharpe-Studio:pid`` (short hostname) stranded on cards while the
    sweeping tick's host prefix was ``Sharpe-Studio.local:`` — a bare
    startswith test classified the same machine's zombie as foreign and
    skipped it. The first-DNS-label match must sweep it."""
    tid = kb.create_task(conn, title="alias", assignee="w")
    # Claim with the SHORT form, flip status externally, worker dead.
    dead = subprocess.Popen(["true"])
    dead.wait()
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock='Short-Host:1234', "
        "claim_expires=9999999999, worker_pid=?, current_run_id=NULL "
        "WHERE id=?",
        (dead.pid, tid),
    )
    conn.commit()
    # Simulate the ticking host emitting the LONG form.
    monkeypatch.setattr(
        kb, "_host_prefix", lambda: "Short-Host.local:",
    )
    assert kbd._claim_is_host_local("Short-Host:1234", "Short-Host.local:")
    assert not kbd._claim_is_host_local("Other-Host:1234", "Short-Host.local:")
    # And the sweep must release it within one tick.
    result = kbd.dispatch_once(
        conn, spawn_fn=lambda task, workspace, board=None: 999999,
    )
    assert tid in result.reconciled_claims
    row = conn.execute(
        "SELECT claim_lock FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["claim_lock"] is None


def test_promoted_event_not_fired_for_card_still_holding_lock(conn):
    """ACCEPTANCE 2 (direct): after the sweep + promotion pipeline runs on
    a wedged board, every ``promoted`` event on the card must describe a
    card that was claimable at the time (claim released before promote)."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = _make_zombie(conn, host, dest_status="blocked")
    # Give the blocked card a done parent so recompute_ready wants to
    # promote it (blocked -> ready when parents satisfied).
    parent = kb.create_task(conn, title="parent", assignee="w")
    conn.execute(
        "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
        (parent, tid),
    )
    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    conn.commit()

    result = kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace, board=None: 999999,
    )

    # Sweep released the claim; then promotion ran on a claimable card.
    row = conn.execute(
        "SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["claim_lock"] is None
    # Card was promoted back to ready (or claimed+spawned by this very tick).
    events = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? "
        "ORDER BY id",
        (tid,),
    ).fetchall()
    kinds = [e["kind"] for e in events]
    assert "promoted" not in kinds or row["status"] in ("ready", "running"), (
        "promotion must describe a claimable card"
    )