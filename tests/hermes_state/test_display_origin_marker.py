"""FC-17: the display projection must MARK rotation rows, never hide them.

In-place compaction archives earlier turns as ``active=0, compacted=1`` rows
and re-sequenced tail clones onto the archived originals' display slots
(the ``messages_display_order_insert`` trigger pins a clone whose content
identity already occupies an earlier display slot, so a carried clone has
``display_order < id`` while a fresh row always owns its slot,
``display_order == id``). Rendered flat, the WebUI showed BOTH the archived
originals and the carried clones as ordinary fresh messages — a rotation
looked like yesterday's conversation re-arriving ("still glitching/repeating").

The fix is metadata, not a filter: ``SessionMessagesMixin._row_to_message_dict``
computes a ``display_origin`` marker per row from existing columns:

- ``"archived"``  — compaction-archived original owning its slot
                    (``active=0, compacted=1, display_order == id``)
- ``"carried"``   — rotation re-sequenced duplicate pinned to an earlier
                    display slot (``display_order < id``), live or archived
- ``"rewound"``   — Undo/Rewind duplicate (``active=0, compacted=0``);
                    display reads exclude these, audit reads mark them
- absent          — fresh live row

These tests pin the two contract invariants: the marker distinguishes the row
classes, and the display projection returns exactly the same rows as before
the marker existed (no rows hidden, no rows added).
"""

import pytest

from hermes_state import SessionDB
from hermes_state_messages import SessionMessagesMixin


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db, sid="s1"):
    """One in-place rotation with a watermark, like the live hot session
    (20260929_202138_ef25e91c): archived originals, pinned tail clones,
    a fresh handoff summary, fresh post-rotation turns, and a rewound pair."""
    db.create_session(sid, source="cli")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "old q1"},
        {"role": "assistant", "content": "old a1"},
    ])
    watermark = db.get_active_message_watermark(sid)
    # Rows that "arrived during the slow summary" (id > watermark): the tail
    # the rotation re-sequences as clones.
    db.append_message(sid, "user", "tail q")
    db.append_message(sid, "assistant", "tail a")
    # tail_count=0: the tail originals are NOT rewind-flagged — they archive as
    # compacted=1 (display-visible), so the trigger PINS the clones onto their
    # display slots: clones have display_order < id (the live-DB signature).
    db.archive_and_compact(sid, [{"role": "assistant", "content": "summary"}],
                           watermark=watermark, tail_count=0)
    # Genuinely fresh post-rotation turns.
    db.append_message(sid, "user", "fresh q")
    db.append_message(sid, "assistant", "fresh a")
    # A sacrificial pair for the Undo/Rewind class (active=0, compacted=0).
    db.append_message(sid, "user", "sacrificial u")
    db.append_message(sid, "assistant", "sacrificial a")
    live = db.get_messages(sid)
    target = next(m for m in reversed(live) if m["content"] == "sacrificial u")
    db.rewind_to_message(sid, target["id"])
    return sid


class TestDisplayOriginMarker:
    def test_marker_distinguishes_archived_carried_and_fresh(self, db):
        """One rotation, three visible row classes, three distinct markers."""
        sid = _seed(db)
        msgs = db.get_messages(sid, include_compacted=True)

        by_marker = {}
        for m in msgs:
            by_marker.setdefault(m.get("display_origin"), []).append(m["content"])

        # Fresh live rows carry NO marker (absent key = fresh).
        assert "fresh q" in by_marker.get(None, []), (
            "fresh post-rotation rows must have no display_origin marker"
        )
        assert "summary" in by_marker.get(None, []), (
            "the handoff summary is a fresh live row and must be unmarked"
        )
        # Carried tail clones are marked "carried" — never plain fresh rows.
        assert "tail q" in by_marker.get("carried", []), (
            "rotation-carried tail clones must be marked display_origin='carried'"
        )
        # Archived originals are marked "archived".
        assert "old q1" in by_marker.get("archived", []), (
            "compaction-archived originals must be marked display_origin='archived'"
        )
        # No row may carry an unexpected marker value.
        assert set(by_marker) <= {None, "archived", "carried"}, (
            f"unexpected display_origin values: {sorted(str(k) for k in by_marker)}"
        )

    def test_rewound_rows_marked_in_audit_reads_only(self, db):
        """Undo/Rewind duplicates stay excluded from display reads; audit reads mark them."""
        sid = _seed(db)
        display = db.get_messages(sid, include_compacted=True)
        assert not any(m.get("display_origin") == "rewound" for m in display), (
            "rewound rows must never surface in the display projection"
        )
        assert not any(m["content"] == "sacrificial u" for m in display), (
            "the rewound pair must stay hidden from the display projection"
        )
        audit = db.get_messages(sid, include_inactive=True)
        assert any(m.get("display_origin") == "rewound" for m in audit), (
            "audit reads must mark Undo/Rewind duplicates display_origin='rewound'"
        )

    def test_carried_marker_derives_from_display_order_pin(self, db):
        """A carried clone is either pinned to an earlier row's display slot
        (display_order < id, raw row check) or an identity clone with a
        lower-id sibling sharing its display identity."""
        sid = _seed(db)
        msgs = db.get_messages(sid, include_compacted=True)
        carried = [m for m in msgs if m.get("display_origin") == "carried"]
        assert carried, "seed must produce carried clones"
        carried_ids = {m["id"] for m in carried}
        raw = {r["id"]: r for r in db._read_all(
            "SELECT id, display_order, display_identity FROM messages WHERE session_id = ?", (sid,))}
        for m in carried:
            r = raw[m["id"]]
            assert r["display_order"] < m["id"] or (
                r["display_identity"] is not None and any(
                    o["display_identity"] == r["display_identity"] and o_id < m["id"]
                    for o_id, o in raw.items() if o_id != m["id"])), (
                "carried rows must be pinned clones or identity clones"
            )
        for m in msgs:
            if m.get("display_origin") is None:
                assert m["active"] == 1, "unmarked rows must be live"

    def test_marker_covers_every_row_state(self):
        """Unit coverage of the marker derivation for all column combinations,
        including the archived-clone state a LATER rotation produces (verified
        on the live hot session: pinned clones archived as compacted=1)."""
        f = SessionMessagesMixin._display_origin
        # fresh live row owning its slot
        assert f({"active": 1, "compacted": 0, "display_order": 9, "id": 9}) is None
        # live clone pinned to an earlier slot
        assert f({"active": 1, "compacted": 0, "display_order": 3, "id": 7}) == "carried"
        # archived original owning its slot
        assert f({"active": 0, "compacted": 1, "display_order": 5, "id": 5}) == "archived"
        # archived clone still pinned (a later rotation archived it)
        assert f({"active": 0, "compacted": 1, "display_order": 3, "id": 7}) == "carried"
        # Undo/Rewind duplicate
        assert f({"active": 0, "compacted": 0, "display_order": 8, "id": 8}) == "rewound"


class TestDisplayProjectionUnchanged:
    def test_marker_adds_no_rows_and_hides_no_rows(self, db):
        """The projection returns the SAME rows as the raw display clause — the
        marker is metadata only. Row-set parity against the SQL projection the
        display read has always used (active=1 OR compacted=1, one row per
        display_order, active preferred)."""
        sid = _seed(db)
        msgs = db.get_messages(sid, include_compacted=True)

        raw = db._read_all(
            """SELECT chosen.* FROM (
                    SELECT display_order, MAX(active) AS best FROM messages
                    WHERE session_id = ? AND (active = 1 OR compacted = 1)
                    GROUP BY display_order
                ) AS page
                JOIN messages AS chosen ON chosen.id = (
                    SELECT candidate.id FROM messages AS candidate
                    WHERE candidate.session_id = ?
                      AND candidate.display_order = page.display_order
                      AND (candidate.active = 1 OR candidate.compacted = 1)
                    ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1
                )
                ORDER BY chosen.display_order ASC""",
            [sid, sid],
        )
        assert [m["id"] for m in msgs] == [r["id"] for r in raw], (
            "display projection row set must be identical with the marker active"
        )

    def test_paging_row_set_unchanged_with_marker(self, db):
        """Paged display reads keep their exact row sets too (page through and
        reassemble)."""
        sid = _seed(db)
        full = db.get_messages(sid, include_compacted=True)
        paged = []
        offset = 0
        while True:
            page = db.get_messages(sid, include_compacted=True, limit=3, offset=offset)
            if not page:
                break
            paged.extend(page)
            offset += 3
        assert [m["id"] for m in paged] == [m["id"] for m in full]