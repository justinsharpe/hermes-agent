"""Replay-diet levers — wire-time trims of replayed history (CONTEXT-DIET-SPEC-v1 §8 Stage 1).

Lever (a): ``tool_calls`` echo trim. Assistant tool-call ``function.arguments`` JSON
echoes replay verbatim on every request; on tool-heavy sessions the §1.3 audit measured
~76k tokens of them in the replayed class. This module trims oversized *historical*
argument payloads on the per-call WIRE COPY only, shrinking long string leaves inside
the parsed JSON so the arguments stay valid (providers 400 on malformed args) and the
same stored row always renders the same trimmed bytes (deterministic → the prompt-cache
prefix breaks once at the config boundary, then stays byte-stable).

Persisted history is never touched: the persisted row keeps the full bytes in
state.db, and the trim re-derives identically on every send. This mirrors the
compressor's own pass-3 trim (``_truncate_tool_call_args_at``) but runs on the send
path so it also applies to rows inside the protected tail that the prune never
rewrites.

Byte-stability rules honored:
- Deterministic: same input → same output bytes, always (no clock, no randomness).
- Idempotent: a trimmed leaf is recognized and never re-trimmed (the #83714
  imitation-marker shape is reused so a model copying the marker verbatim stays
  visibly wrong and stale).
- Off by default (``replay_diet.tool_calls_max_args_chars: 0``): unset config is
  byte-identical to pre-feature behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolved ``replay_diet.tool_calls_max_args_chars`` per profile home (the multiplexed
# gateway serves every profile from one process — a single slot would hand the launch
# profile's diet to every other profile). 0 = off. Mirrors tools/tool_output_limits.py.
_cached_limits: Dict[str, int] = {}
DEFAULT_TOOL_CALLS_MAX_ARGS_CHARS = 0  # off: unset config is byte-identical to pre-feature


def get_replay_diet_limits() -> int:
    """Resolved ``tool_calls_max_args_chars``; never raises. 0 = lever off."""
    from hermes_constants import hermes_home_key

    key = hermes_home_key()
    cached = _cached_limits.get(key)
    if cached is not None:
        return cached
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        section = cfg.get("replay_diet") if isinstance(cfg, dict) else None
        raw = section.get("tool_calls_max_args_chars") if isinstance(section, dict) else None
    except Exception:
        raw = None
    try:
        value = max(0, int(raw)) if raw is not None else DEFAULT_TOOL_CALLS_MAX_ARGS_CHARS
    except (TypeError, ValueError):
        value = DEFAULT_TOOL_CALLS_MAX_ARGS_CHARS
    _cached_limits[key] = value
    return value


def _reset_replay_diet_limits_cache() -> None:
    """Reset the cached limits — for tests or after config hot-reload."""
    _cached_limits.clear()

# Reuse the compressor's imitation-safe truncation marker machinery (single owner of
# the marker shape; a second marker family would let a model imitate one but not the
# other). Lazy import: agent.context_compressor is a heavy module.
def _truncate_args_json(args: str, head_chars: int) -> Optional[str]:
    """Trim long string leaves inside a tool-call arguments JSON blob, keeping it valid.

    Returns ``None`` when nothing was changed (caller keeps the original string —
    re-serialising unchanged bytes would rewrite the wire form and count as phantom
    pressure). Never raises: a malformed historical blob is returned unchanged.
    """
    try:
        from agent.context_compressor import _truncate_tool_call_args_json
    except Exception:  # pragma: no cover - import guard
        return None
    if not isinstance(args, str) or len(args) <= head_chars + 512:
        # Cheap pre-gate: the marker itself is ~180 chars; below this the trim can
        # never reclaim enough to be worth parsing the JSON.
        return None
    trimmed = _truncate_tool_call_args_json(args, head_chars=head_chars)
    return trimmed if trimmed != args else None


def _trim_tool_calls_in_place(api_msg: Dict[str, Any], max_args_chars: int) -> bool:
    """Trim oversized tool_call argument payloads on ONE wire-copy message, in place.

    Returns True when any payload was trimmed. Copy-on-write per tool_call entry so a
    caller passing shallow copies never writes into the persisted turn (same defense
    as ``_canonicalize_api_tool_calls``).
    """
    tool_calls = api_msg.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return False
    changed = False
    new_tcs: List[Any] = []
    for tc in tool_calls:
        if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
            args = tc["function"].get("arguments")
            if isinstance(args, str):
                trimmed = _truncate_args_json(args, max_args_chars)
                if trimmed is not None:
                    tc = {**tc, "function": {**tc["function"], "arguments": trimmed}}
                    changed = True
        new_tcs.append(tc)
    if changed:
        api_msg["tool_calls"] = new_tcs
    return changed


def trim_tool_calls_echoes(
    api_messages: List[Dict[str, Any]], max_args_chars: int, *, history_end_idx: Optional[int] = None,
) -> int:
    """Trim oversized historical tool_call argument echoes on the wire copy.

    Only rows BEFORE the current turn's live rows are eligible (``history_end_idx`` is
    the exclusive end of the replayed prefix; ``None`` = every assistant row). Rows
    appended by the current turn are live tool calls whose exact arguments the model
    just chose and may still be iterating on — trimming those mid-turn breaks the
    tool/result pairing contract the loop relies on for dedup and repair.

    Returns the number of messages trimmed. Zero config (``max_args_chars <= 0``) is a
    byte-identical no-op.
    """
    if max_args_chars is None or max_args_chars <= 0 or not api_messages:
        return 0
    history_end = len(api_messages) if history_end_idx is None else max(0, min(int(history_end_idx), len(api_messages)))
    trimmed = 0
    try:
        for idx in range(history_end):
            msg = api_messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "assistant" and _trim_tool_calls_in_place(msg, int(max_args_chars)):
                trimmed += 1
    except Exception:  # noqa: BLE001 — a diet lever must never kill the send path
        logger.debug("replay-diet tool_calls echo trim failed; sending untrimmed", exc_info=True)
        return 0
    if trimmed and not getattr(logging.getLogger("agent.replay_diet"), "disabled", False):
        logger.debug("replay-diet: trimmed tool_calls argument echoes on %d historical message(s)", trimmed)
    return trimmed


__all__ = [
    "trim_tool_calls_echoes",
    "get_replay_diet_limits",
    "DEFAULT_TOOL_CALLS_MAX_ARGS_CHARS",
]