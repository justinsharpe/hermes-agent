"""Scratch-dir retention: idle detection plus the reaping an idle tree needs before it goes.

Idle entries are not deleted outright anymore: they are *quarantined* — moved to a sibling
``<scratch>-quarantine/`` directory for ``HERMES_SCRATCH_QUARANTINE_HOURS`` (default 72) first.
A multi-day lane that parked a deliverable in scratch loses hours, not the work, and every
departure is logged to ``<scratch>-prune.log`` beside the root so silent sweeps cannot happen.
An entry holding a ``.scratch-keep`` file is exempt entirely — the one-file contract for
agents protecting work.

Deleting a quarantined entry still needs the reaping: a lane's e2e run leaves headless
browsers whose cwd was inside the tree (they survived for days with a ``(deleted)`` cwd),
and a repo whose linked worktree lived in the tree keeps a dangling registration until
someone runs ``git worktree prune``. Expired entries are reaped and released right before
their ``rmtree``.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# How long a TERMed process gets before KILL; browsers exit well within this.
_REAP_GRACE_SECONDS = 3.0
# ``.git`` files (linked worktrees) are looked for this deep; lanes nest repo/tree/subtree.
_GIT_FILE_MAX_DEPTH = 4
# An entry carrying this marker file anywhere in its top level is never pruned: the
# one-file contract for lanes that need a multi-day parking spot in scratch.
SCRATCH_KEEP_MARKER = ".scratch-keep"
# Quarantined entries live this long before final deletion (env-overridable).
_QUARANTINE_HOURS_DEFAULT = 72.0


def _quarantine_hours() -> float:
    """``HERMES_SCRATCH_QUARANTINE_HOURS`` when set and parseable, else 72."""
    raw = os.environ.get("HERMES_SCRATCH_QUARANTINE_HOURS", "").strip()
    try:
        hours = float(raw)
        if hours >= 0:
            return hours
    except ValueError:
        pass
    return _QUARANTINE_HOURS_DEFAULT


def _entry_is_kept(entry: Path) -> bool:
    """True when the entry opts out of pruning via the keep-marker contract."""
    try:
        return (entry / SCRATCH_KEEP_MARKER).exists()
    except OSError:
        return False  # unreadable: not proof of keep; the idle test decides


def _prune_log_path(scratch_root: Path) -> Path:
    """``<scratch>/../scratch-prune.log`` lives beside the scratch root, outside the sweep."""
    return scratch_root.parent / "scratch-prune.log"


def _log_departure(scratch_root: Path, action: str, entry: Path, note: str = "", size: int | None = None) -> None:
    """Append one line per pruned/quarantined entry; never raises into the prune path."""
    try:
        if size is None:
            size = _tree_bytes(entry)
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} action={action} entry={entry.name!r} bytes={size}{(' ' + note) if note else ''}\n"
        with open(_prune_log_path(scratch_root), "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass  # logging must never break the prune


def _tree_bytes(entry: Path) -> int:
    """Total bytes under *entry* (files only); 0 on any error — a log nicety, not a guarantee."""
    total = 0
    stack = [str(entry)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        if child.is_dir(follow_symlinks=False):
                            stack.append(child.path)
                        else:
                            total += child.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def subtree_touched_since(path: Path, cutoff: float) -> bool:
    """True when *path* or anything beneath it has an mtime at or after *cutoff*.

    Stops at the first recent entry, so a live tree costs one hit and only a truly idle
    tree pays for the full walk (once, right before it is deleted). Symlinks are never
    followed: a link into the repo would make the target's activity keep the entry alive.
    An unreadable entry is kept: an incomplete scan cannot establish that it is idle.
    """
    try:
        if os.lstat(path).st_mtime >= cutoff:
            return True
        if not path.is_dir() or path.is_symlink():
            return False
    except OSError:
        return True
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for child in it:
                    try:
                        if child.stat(follow_symlinks=False).st_mtime >= cutoff:
                            return True
                    except OSError:
                        return True
                    if child.is_dir(follow_symlinks=False):
                        stack.append(child.path)
        except OSError:
            return True
    return False


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _own_lineage() -> set[int]:
    """This process and its ancestors: never reap the shell that is running the prune."""
    import psutil

    pids: set[int] = set()
    try:
        proc = psutil.Process()
        while proc is not None and proc.pid not in pids:
            pids.add(proc.pid)
            proc = proc.parent()
    except (psutil.Error, OSError):
        pass
    return pids


def reap_processes_rooted_in(scratch_root: Path, doomed: list[Path]) -> int:
    """TERM (then KILL) same-user processes whose cwd is inside an entry about to be
    pruned, or is a path under *scratch_root* that no longer exists. Returns the count.

    Only the cwd is consulted: a process merely holding a file open inside scratch (an
    editor, a log tail) is not ours to kill, but one *living* in a directory we are
    about to delete, or in one already gone, has nothing left to run for.
    """
    import psutil

    root = os.path.realpath(str(scratch_root))
    targets = [os.path.realpath(str(p)) for p in doomed]
    skip = _own_lineage()
    uid = os.getuid() if hasattr(os, "getuid") else None
    victims: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid"]):
        if proc.pid in skip:
            continue
        try:
            if uid is not None and proc.uids().real != uid:
                continue
            cwd = proc.cwd()
        except (psutil.Error, OSError):
            continue
        if not cwd:
            continue
        deleted = cwd.endswith(" (deleted)")
        cwd_path = cwd[: -len(" (deleted)")] if deleted else cwd
        if not _under(cwd_path, root):
            continue
        if deleted or not os.path.exists(cwd_path) or any(_under(cwd_path, t) for t in targets):
            victims.append(proc)
    if not victims:
        return 0
    for proc in victims:
        try:
            proc.terminate()
        except (psutil.Error, OSError):
            continue
    _, alive = psutil.wait_procs(victims, timeout=_REAP_GRACE_SECONDS)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.Error, OSError):
            continue
    logger.info("scratch prune: reaped %d process(es) rooted in pruned entries", len(victims))
    return len(victims)


def _linked_worktree_repos(entry: Path) -> set[str]:
    """Repos whose linked worktrees live inside *entry* (``.git`` FILES, ``gitdir: <repo>/.git/worktrees/<n>``)."""
    repos: set[str] = set()
    stack = [(str(entry), 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as it:
                for child in it:
                    if child.name == ".git" and child.is_file(follow_symlinks=False):
                        try:
                            line = Path(child.path).read_text(encoding="utf-8", errors="replace").strip()
                        except OSError:
                            continue
                        if line.startswith("gitdir:"):
                            gitdir = Path(line[len("gitdir:"):].strip())
                            # <repo>/.git/worktrees/<name> -> <repo>
                            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                                repos.add(str(gitdir.parent.parent.parent))
                    elif child.is_dir(follow_symlinks=False) and depth < _GIT_FILE_MAX_DEPTH \
                            and child.name not in ("node_modules", ".venv", "venv"):
                        stack.append((child.path, depth + 1))
        except OSError:
            continue
    return repos


def release_git_worktrees(repos: set[str]) -> None:
    """``git worktree prune`` in each repo: drops registrations whose tree we just deleted."""
    for repo in sorted(repos):
        if not os.path.isdir(repo):
            continue
        try:
            subprocess.run(
                ["git", "-C", repo, "worktree", "prune"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("git worktree prune in %s failed: %s", repo, exc)


def prune_idle_entries(root: Path, max_idle_hours: float, skip_names: frozenset[str]) -> int:
    """Quarantine top-level entries of *root* with no write anywhere in their subtree for
    *max_idle_hours*, and finally delete quarantined entries once their grace expires.
    Keep-marked entries (``.scratch-keep``) are never touched. Every departure is logged
    to ``<root>/../scratch-prune.log``. Returns the count of entries that left the root
    (quarantined this pass + deleted from quarantine this pass)."""
    cutoff = time.time() - max_idle_hours * 3600
    quarantine = root.parent / (root.name + "-quarantine")
    quarantine.mkdir(parents=True, exist_ok=True)
    try:
        entries = [e for e in root.iterdir() if e.name not in skip_names]
    except OSError:
        return 0
    doomed = [e for e in entries if not subtree_touched_since(e, cutoff) and not _entry_is_kept(e)]
    # Runs even with nothing to delete: orphans whose cwd was removed by an earlier pass
    # (or by hand) are found by the deleted-cwd rule, not by membership in ``doomed``.
    try:
        reap_processes_rooted_in(root, doomed)
    except Exception as exc:  # psutil missing or restricted host: the deletion still proceeds
        logger.debug("scratch prune: process reap skipped: %s", exc)
    moved = 0
    for entry in doomed:
        target = quarantine / entry.name
        if target.exists():
            # A previous pass already quarantined this name: keep both by suffixing the move.
            target = quarantine / f"{entry.name}.prior-{int(time.time())}"
        try:
            size = _tree_bytes(entry)  # before the move: the source path is gone afterwards
            shutil.move(str(entry), str(target))
            _stamp_quarantine(target)
            moved += 1
            _log_departure(root, "quarantined", entry, f"grace_hours={_quarantine_hours():g} to={target.name}", size=size)
        except OSError as exc:
            logger.debug("scratch prune: quarantine move failed for %s: %s", entry, exc)
            continue
    # Final deletion for entries whose grace expired while in quarantine.
    grace_seconds = _quarantine_hours() * 3600
    removed = 0
    try:
        expired = [e for e in quarantine.iterdir() if _quarantined_expired(e, grace_seconds)]
    except OSError:
        expired = []
    for entry in expired:
        repos: set[str] = set()
        try:
            if entry.is_dir() and not entry.is_symlink():
                repos = _linked_worktree_repos(entry)
                reap_processes_rooted_in(quarantine, [entry])
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
            removed += 1
            _log_departure(root, "deleted", entry, "source=quarantine")
            release_git_worktrees(repos)
        except OSError:
            continue
    return moved + removed


_QUARANTINE_STAMP = ".quarantined-at"


def _stamp_quarantine(entry: Path) -> None:
    """Anchor the grace window at quarantine time. Directories carry a stamp file inside;
    a top-level *file* entry cannot, so its mtime is reset to now instead (the expiry
    fallback reads subtree mtimes, so the anchor is the same either way)."""
    try:
        if entry.is_dir() and not entry.is_symlink():
            (entry / _QUARANTINE_STAMP).write_text(f"{time.time()}\n", encoding="utf-8")
        else:
            os.utime(entry, None)
    except OSError:
        pass


def _quarantined_expired(entry: Path, grace_seconds: float) -> bool:
    """True when a quarantined entry's grace window has elapsed. The stamp written at
    quarantine time is the start of grace; an entry without one (moved by hand) falls
    back to whole-subtree idleness, and a keep-marker exempts it from final deletion."""
    if _entry_is_kept(entry):
        return False
    stamp = entry / _QUARANTINE_STAMP
    try:
        quarantined_at = float(stamp.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return not subtree_touched_since(entry, time.time() - grace_seconds)
    return (time.time() - quarantined_at) >= grace_seconds
