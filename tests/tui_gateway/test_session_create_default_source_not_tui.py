"""Regression: session.create without source must NOT default to "tui".

Before Fix A (ADR t_43ccd597), a WS client that called ``session.create`` without
passing an explicit ``source`` field would have its session stamped ``source="tui"``
— even when Hermes was running headless (``hermes serve`` without ``HERMES_DESKTOP``).

``source="tui"`` is wrong on two counts:
1. A headless ``hermes serve`` has no terminal attached, so "tui" is misleading.
2. The REST allowlist (Fix B) did not include "tui", so persisted ``tui`` rows
   were invisible to ``GET /api/sessions`` and ``GET /api/sessions/{id}/messages``.

Fix A changes the layered default so that, when no ``source`` is passed and
``HERMES_DESKTOP`` is unset, the resolved source is ``"desktop"`` (the only live
controller on a headless serve) instead of ``"tui"``.

This test documents the fix and guards against regression.
"""

import pytest


def _reload_resolver():
    # Plain import — every resolver under test reads the env at CALL time, so
    # no reload is needed.
    import tui_gateway.server as _srv
    return _srv


@pytest.fixture
def clean_env(monkeypatch):
    """Hermes running headless: no HERMES_DESKTOP, no HERMES_DESKTOP_TERMINAL."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    return monkeypatch


class TestSessionCreateDefaultSourceNotTui:
    """session.create without source must default to 'desktop', not 'tui'."""

    def test_no_source_no_hermes_desktop_resolves_to_desktop(self, clean_env):
        """The core regression: a headless serve must NOT stamp sessions as 'tui'."""
        _srv = _reload_resolver()
        assert _srv._resolve_session_source(None) == "desktop"

    def test_no_source_with_hermes_desktop_still_resolves_to_desktop(self, clean_env):
        """HERMES_DESKTOP=1 without HERMES_DESKTOP_TERMINAL → 'desktop' (unchanged)."""
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_source(None) == "desktop"

    def test_explicit_source_wins_even_when_hermes_desktop_set(self, clean_env):
        """An explicit source must always be preserved, regardless of env vars."""
        clean_env.setenv("HERMES_DESKTOP", "1")
        _srv = _reload_resolver()
        assert _srv._resolve_session_source("telegram") == "telegram"
        assert _srv._resolve_session_source("mobile") == "mobile"
        assert _srv._resolve_session_source("webchat") == "webchat"

    def test_tui_is_still_reserved_for_embedded_terminal(self, clean_env):
        """HERMES_DESKTOP=1 WITH HERMES_DESKTOP_TERMINAL=1 → 'tui' (back-compat)."""
        clean_env.setenv("HERMES_DESKTOP", "1")
        clean_env.setenv("HERMES_DESKTOP_TERMINAL", "true")
        _srv = _reload_resolver()
        assert _srv._resolve_session_source(None) == "tui"
