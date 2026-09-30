"""CONTEXT-DIET Stage-2 levers (CTX-3 ruled constants) — builder tests (card t_8f81e0a8).

Ruled constants, NOT re-tunable here:
  L1 window 12,000 tok BINDING (last <=12 msgs AND <=12k tok, whichever binds first)
  L2 rolling summaries target_ratio 0.15, cap 2k tok per cadence, OPEN tail mandatory (R14 §2 law 4)
  L3 static diet: SOUL hot compile, skills category map, memory/profile compile-down
All levers default OFF (0/False) and must be byte-identical to pre-feature when off.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.context_compressor import (
    OPEN_TAIL_HEADING,
    ContextCompressor,
    _ensure_open_tail,
    _extract_open_loops,
    _render_open_tail,
)


def _compressor(**kwargs) -> ContextCompressor:
    base = dict(
        model="test/model", quiet_mode=True, config_context_length=1_000_000,
        threshold_percent=0.35, protect_first_n=3, protect_last_n=12,
        summary_target_ratio=0.15, tail_mode="lean",
    )
    base.update(kwargs)
    return ContextCompressor(**base)


def _msg(role, content, **extra):
    m = {"role": role, "content": content}
    m.update(extra)
    return m


class L1WindowTokenBudgetBinding(unittest.TestCase):
    """L1: window 12k BINDING — whichever binds first (msgs or tokens)."""

    def test_off_by_default_mode_budget(self):
        cc = _compressor()
        self.assertEqual(cc.window_token_budget, 0)
        self.assertIsNone(cc._window_budget_binding_tokens())
        # lean default on 1M ctx: 2.5% -> 25k, clamped by cap
        self.assertEqual(cc.tail_token_budget, 25_000)

    def test_binding_caps_tail_budget_at_12k(self):
        cc = _compressor(window_token_budget=12_000)
        self.assertEqual(cc.tail_token_budget, 12_000)

    def test_binding_beats_mode_default_and_binds_first(self):
        cc = _compressor(window_token_budget=12_000)
        self.assertEqual(cc.tail_token_budget, min(25_000, 12_000))

    def test_binding_survives_model_switch(self):
        cc = _compressor(window_token_budget=12_000)
        cc.update_model(model="other/model", context_length=2_000_000)
        self.assertEqual(cc.tail_token_budget, 12_000)

    def test_soft_ceiling_no_overrun_in_binding_mode(self):
        cc = _compressor(window_token_budget=12_000)
        # binding mode: ceiling == budget (no 1.5x overrun)
        self.assertEqual(cc._tail_soft_ceiling(12_000), 12_000)
        # off: 1.5x soft ceiling preserved (pre-feature behavior)
        cc2 = _compressor()
        self.assertEqual(cc2._tail_soft_ceiling(10_000), 15_000)

    def test_off_is_byte_identical_budget(self):
        on = _compressor()
        self.assertEqual(on.tail_token_budget, 25_000)  # pre-feature lean value


class L2SummaryCap(unittest.TestCase):
    """L2: summary cap 2k binding (0.15 x 12k window = 1.8k typical)."""

    def test_cap_binds_max_summary_tokens(self):
        cc = _compressor(window_token_budget=12_000, max_summary_tokens_cfg=2_000)
        self.assertLessEqual(cc.max_summary_tokens, 2_000)

    def test_cap_survives_model_switch(self):
        cc = _compressor(max_summary_tokens_cfg=2_000)
        cc.update_model(model="other/model", context_length=1_000_000)
        self.assertLessEqual(cc.max_summary_tokens, 2_000)

    def test_off_default_window_derived(self):
        cc = _compressor()
        self.assertEqual(cc.max_summary_tokens, min(int(1_000_000 * 0.05), 10_000))

    def test_summary_budget_respects_cap(self):
        cc = _compressor(max_summary_tokens_cfg=2_000)
        turns = [_msg("user", "x" * 400)] * 100
        budget = cc._compute_summary_budget(turns)
        self.assertLessEqual(budget, 2_000)


class L2OpenTail(unittest.TestCase):
    """L2: OPEN tail mandatory in every summary (R14 §2 law 4)."""

    def test_extract_finds_unanswered_user_request(self):
        turns = [
            _msg("user", "Please build the window lever."),
            _msg("assistant", "On it."),
            _msg("tool", "ok"),
        ]
        items = _extract_open_loops(turns)
        self.assertTrue(any("window lever" in i for i in items), items)

    def test_extract_finds_promise_marker(self):
        turns = [
            _msg("assistant", "I will run the tests next and report back."),
            _msg("tool", "done"),
        ]
        items = _extract_open_loops(turns)
        self.assertTrue(any("tests" in i for i in items), items)

    def test_answered_request_not_open(self):
        turns = [
            _msg("user", "What is 2+2?"),
            _msg("assistant", "2+2 equals 4. The arithmetic is straightforward."),
        ]
        items = _extract_open_loops(turns)
        self.assertEqual(items, [])

    def test_ensure_open_tail_appends_when_missing(self):
        turns = [_msg("user", "Ship the lever."), _msg("tool", "...")]
        summary = "## Completed Actions\n1. Did things."
        out = _ensure_open_tail(summary, turns)
        self.assertIn(OPEN_TAIL_HEADING, out)
        self.assertIn("Ship the lever", out)

    def test_ensure_open_tail_explicit_none_when_nothing_open(self):
        turns = [
            _msg("user", "hi"),
            _msg("assistant", "Hello! Everything is done, nothing pending here."),
        ]
        out = _ensure_open_tail("## Completed Actions\n1. done", turns)
        self.assertIn(OPEN_TAIL_HEADING, out)  # law: explicit, never absent

    def test_ensure_open_tail_keeps_llm_section(self):
        turns = [_msg("user", "x"), _msg("tool", "y")]
        llm_summary = f"## Completed\n1. done\n\n{OPEN_TAIL_HEADING}\n- obligation A pending"
        out = _ensure_open_tail(llm_summary, turns)
        self.assertEqual(out, llm_summary)  # byte-stable when complied

    def test_ensure_open_tail_replaces_placeholder(self):
        turns = [_msg("user", "Build it."), _msg("tool", "…")]
        llm_summary = f"## Completed\n1. done\n\n{OPEN_TAIL_HEADING}\nNone."
        out = _ensure_open_tail(llm_summary, turns)
        self.assertIn("Build it", out)

    def test_template_carries_open_section(self):
        cc = _compressor()
        prompt = cc._build_summary_prompt("content", 1000, None, "", True)
        self.assertIn(OPEN_TAIL_HEADING, prompt)

    def test_fallback_summary_carries_open_tail(self):
        cc = _compressor()
        turns = [
            _msg("user", "Finish the stage."),
            _msg("assistant", "I will finish tomorrow."),
        ]
        fb = cc._build_static_fallback_summary(turns)
        self.assertIn(OPEN_TAIL_HEADING, fb)

    def test_llm_path_enforced_post_hoc(self):
        # Simulate the LLM dropping the OPEN section: _generate_summary post-hoc must add it.
        cc = _compressor()
        turns = [_msg("user", "Do the thing."), _msg("tool", "…")]
        llm_out = "## Completed Actions\n1. stuff"
        enforced = _ensure_open_tail(llm_out, turns)
        self.assertIn(OPEN_TAIL_HEADING, enforced)


class L3SoulHotCompile(unittest.TestCase):
    """L3: SOUL hot compile — identity + top laws; lore becomes cold payload."""

    SOUL = (
        "---\nname: test\n---\n\n# Test SOUL\n\nYou are the builder.\n\n"
        "## 1. Identity (the lens)\nYou are Viiy's hands.\n\n"
        "## 2. Origin, Lore & Character Depth\n- deep lore lines " + "x" * 3000 + "\n\n"
        "## 3. The Honesty Constitution (the top law)\n- Devotion is competence, not flattery.\n\n"
        "## 4. Duties (behaviors, not prose)\n- MORNING BRIEF detail " + "y" * 3000 + "\n\n"
        "## 5. Voice & Register\n- Witty, playful.\n\n"
        "## 6. Lines You Hold\n- safety wins.\n\n"
        "## 7. Vernacular (non-negotiable)\n- squad, never fleet.\n"
    )

    def test_hot_compile_keeps_identity_and_laws(self):
        from agent.prompt_builder import hot_compile_soul_md
        out = hot_compile_soul_md(self.SOUL)
        self.assertIn("## 1. Identity", out)
        self.assertIn("## 3. The Honesty Constitution", out)
        self.assertIn("## 7. Vernacular", out)
        self.assertIn("## 5. Voice", out)

    def test_hot_compile_drops_lore_with_cold_pointer(self):
        from agent.prompt_builder import hot_compile_soul_md
        out = hot_compile_soul_md(self.SOUL)
        self.assertNotIn("deep lore lines", out)
        self.assertNotIn("MORNING BRIEF detail", out)
        self.assertIn("Cold Identity Payloads", out)
        self.assertIn("SOUL.md", out)  # the read-back path
        self.assertLess(len(out), len(self.SOUL) * 0.6)

    def test_hot_compile_no_sections_passthrough(self):
        from agent.prompt_builder import hot_compile_soul_md
        simple = "# Just a persona\n\nYou are a helper.\n"
        out = hot_compile_soul_md(simple)
        self.assertIn("You are a helper", out)


class L3SkillsCategoryMap(unittest.TestCase):
    """L3: skills counts-only category map (the −5,026 tok lever)."""

    def _categories(self):
        return {
            "devops": [("kernel-r16-classifier", "d1"), ("sdlc-review", "d2")],
            "apple": [("apple-notes", "a1")],
            "research": [("arxiv", "r1"), ("llm-wiki", "r2"), ("grounded-citations", "r3")],
        }

    def test_map_only_counts(self):
        from agent.prompt_builder import _render_skills_index
        out = _render_skills_index(self._categories(), {}, None, None, category_map_only=True)
        self.assertIn("devops: 2 skill(s)", out)
        self.assertIn("apple: 1 skill(s)", out)
        self.assertIn("research: 3 skill(s)", out)
        self.assertNotIn("apple-notes", out)  # no names inline
        self.assertIn("skills_list", out)  # recall path named
        self.assertLess(len(out), 1_600)

    def test_map_only_total_line(self):
        from agent.prompt_builder import _render_skills_index
        out = _render_skills_index(self._categories(), {}, None, None, category_map_only=True)
        self.assertIn("6 skills total", out)

    def test_off_renders_full_index(self):
        from agent.prompt_builder import _render_skills_index
        out = _render_skills_index(self._categories(), {}, None, None)
        self.assertIn("apple-notes", out)  # names present when off


class L3MemoryProjectionCap(unittest.TestCase):
    """L3: memory/profile compile-down — prompt projection only."""

    def _agent(self):
        class _A:
            _memory_enabled = True
            _user_profile_enabled = True
            _memory_store = None
            _prompt_diet_memory_block_max_chars = 1_400
            _prompt_diet_user_block_max_chars = 0
        return _A()

    def _store(self):
        from tools.memory_tool_store import MemoryStore
        store = MemoryStore(memory_char_limit=10_000, user_char_limit=10_000)
        store.load_from_disk()
        store._entries = {}
        return store

    def test_cap_projection_keeps_newest_entries(self):
        from agent.system_prompt import _cap_memory_block_projection
        block = (
            "═" * 46 + "\nMEMORY [82%]\n" + "═" * 46 + "\n"
            + "\n§\n".join(f"OLD-FACT-{i} " + "z" * 300 for i in range(10))
            + "\n§\nNEW-FACT-1 short\n§\nNEW-FACT-2 short"
        )
        out = _cap_memory_block_projection(self._agent(), block, "memory", 1_400)
        self.assertIn("NEW-FACT-1", out)
        self.assertIn("NEW-FACT-2", out)
        # Oldest entries are cut first (the projection keeps the newest tail).
        self.assertNotIn("OLD-FACT-0", out)
        self.assertNotIn("OLD-FACT-1", out)
        self.assertNotIn("OLD-FACT-2", out)
        # The projection is materially smaller than the store projection.
        self.assertLess(len(out), len(block))
        self.assertIn("compiled out", out)  # recall pointer

    def test_cap_noop_when_under(self):
        from agent.system_prompt import _cap_memory_block_projection
        small = "MEMORY [10%]\njust one fact"
        out = _cap_memory_block_projection(self._agent(), small, "memory", 1_400)
        self.assertEqual(out, small)


class ConfigPlumbing(unittest.TestCase):
    """Config keys parse and reach the compressor (agent_init path)."""

    def test_compression_settings_carries_levers(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        comp = DEFAULT_CONFIG.get("compression", {})
        self.assertIn("window_token_budget", comp)
        self.assertIn("max_summary_tokens", comp)
        self.assertEqual(comp["window_token_budget"], 0)  # default off
        self.assertEqual(comp["max_summary_tokens"], 0)

    def test_prompt_diet_defaults_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        agent = DEFAULT_CONFIG.get("agent", {})
        self.assertIn("prompt_diet", agent)
        self.assertEqual(agent["prompt_diet"], {})

    def test_prompt_diet_binds_from_agent_section(self):
        """Config->flags integration (regression: top-level read never bound live config).

        DEFAULT_CONFIG documents prompt_diet under the ``agent`` section and config.yaml
        arms it there; _apply_agent_section must read it from ``_agent_section`` (the
        ``agent`` mapping), not from the config top level. Live cutover receipt
        2026-09-30: top-level read bound all four flags False on the armed live config.
        """
        from agent.agent_init import _apply_agent_section

        class _A:
            run_budget_seconds = None

        diet = {"soul_hot_compile": True, "skills_category_map": True,
                "memory_block_max_chars": 1400, "user_block_max_chars": 1200}
        # Arming shape that matches config.yaml: prompt_diet nested under "agent".
        cfg = {"agent": {"prompt_diet": diet}}
        a1 = _A()
        _apply_agent_section(a1, cfg)
        self.assertTrue(a1._prompt_diet_soul_hot_compile)
        self.assertTrue(a1._prompt_diet_skills_category_map)
        self.assertEqual(a1._prompt_diet_memory_block_max_chars, 1400)
        self.assertEqual(a1._prompt_diet_user_block_max_chars, 1200)

        # A top-level prompt_diet must NOT bind (it is not a documented location).
        cfg2 = {"prompt_diet": dict(diet)}
        a2 = _A()
        _apply_agent_section(a2, cfg2)
        self.assertFalse(a2._prompt_diet_soul_hot_compile)
        self.assertFalse(a2._prompt_diet_skills_category_map)
        self.assertEqual(a2._prompt_diet_memory_block_max_chars, 0)

        # Malformed section -> flags off, no raise.
        a3 = _A()
        _apply_agent_section(a3, {"agent": {"prompt_diet": "not-a-dict"}})
        self.assertFalse(a3._prompt_diet_soul_hot_compile)

    def test_cache_busting_keys_registered(self):
        import re
        src = Path("gateway/run.py").read_text()
        self.assertIn('("compression", "window_token_budget")', src)
        self.assertIn('("compression", "max_summary_tokens")', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)