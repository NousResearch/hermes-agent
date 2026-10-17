"""Tests for the configurable busy-ack template helper (issue #26024).

Covers the unit-level surface in :mod:`tools.busy_ack_templates`:

  * Defaults exactly match the hardcoded strings that shipped before this
    feature -- guards against an accidental wording drift breaking
    deployments that haven't opted into overrides.
  * Resolver correctness for every degenerate input shape (None, empty,
    wrong type, missing mode, non-string override).
  * Render correctness with and without status detail.
  * Empty / whitespace override is honoured (operator can suppress one
    mode without disabling busy-ack globally).
  * Env-var roundtrip drops malformed entries.
  * Unknown placeholders in a user-supplied template don't crash the
    render path.
"""

from __future__ import annotations

import json

import pytest

from tools.busy_ack_templates import (
    DEFAULT_TEMPLATES,
    ENV_VAR_TEMPLATES,
    VALID_MODES,
    encode_templates_for_env,
    load_templates_from_env,
    render_busy_ack,
    resolve_busy_ack_template,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaultsMatchHistoricalStrings:
    """If any of these change, every deployment that hasn't overridden
    templates sees a behaviour drift. Re-enabling these requires explicit
    coordination with downstream operators."""

    def test_interrupt_default_unchanged(self):
        assert DEFAULT_TEMPLATES["interrupt"] == (
            "⚡ Interrupting current task{status_detail}. "
            "I'll respond to your message shortly."
        )

    def test_queue_default_unchanged(self):
        assert DEFAULT_TEMPLATES["queue"] == (
            "⏳ Queued for the next turn{status_detail}. "
            "I'll respond once the current task finishes."
        )

    def test_steer_default_unchanged(self):
        assert DEFAULT_TEMPLATES["steer"] == (
            "⏩ Steered into current run{status_detail}. "
            "Your message arrives after the next tool call."
        )

    def test_valid_modes_match_default_keys(self):
        assert VALID_MODES == set(DEFAULT_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# resolve_busy_ack_template
# ---------------------------------------------------------------------------


class TestResolveBusyAckTemplate:
    @pytest.mark.parametrize("mode", ["interrupt", "queue", "steer"])
    def test_none_templates_returns_default(self, mode):
        assert resolve_busy_ack_template(None, mode) == DEFAULT_TEMPLATES[mode]

    @pytest.mark.parametrize("mode", ["interrupt", "queue", "steer"])
    def test_empty_templates_returns_default(self, mode):
        assert resolve_busy_ack_template({}, mode) == DEFAULT_TEMPLATES[mode]

    def test_partial_templates_falls_back_per_mode(self):
        """Override one mode, default the others."""
        templates = {"queue": "Custom queue {status_detail}."}
        assert resolve_busy_ack_template(templates, "queue") == "Custom queue {status_detail}."
        assert resolve_busy_ack_template(templates, "interrupt") == DEFAULT_TEMPLATES["interrupt"]
        assert resolve_busy_ack_template(templates, "steer") == DEFAULT_TEMPLATES["steer"]

    def test_unknown_mode_falls_back_to_interrupt(self):
        """Defensive: any unknown mode (e.g. typo) uses the interrupt default
        -- matches the runtime fallback in gateway.run."""
        assert resolve_busy_ack_template(None, "BOGUS") == DEFAULT_TEMPLATES["interrupt"]
        assert resolve_busy_ack_template({"queue": "X"}, "") == DEFAULT_TEMPLATES["interrupt"]

    def test_non_mapping_templates_falls_back(self, caplog):
        """A malformed top-level value (list, string, etc.) must not crash."""
        # YAML 1.1 could parse `display.busy_ack_templates: ""` as an empty
        # string. Make sure that degrades gracefully.
        assert resolve_busy_ack_template("not-a-dict", "queue") == DEFAULT_TEMPLATES["queue"]
        assert resolve_busy_ack_template(["queue"], "queue") == DEFAULT_TEMPLATES["queue"]

    def test_non_string_override_falls_back(self):
        """If a YAML file sets ``queue: 42`` we must not crash mid-busy-turn."""
        templates = {"queue": 42, "steer": ["a", "b"], "interrupt": None}
        assert resolve_busy_ack_template(templates, "queue") == DEFAULT_TEMPLATES["queue"]
        assert resolve_busy_ack_template(templates, "steer") == DEFAULT_TEMPLATES["steer"]
        assert resolve_busy_ack_template(templates, "interrupt") == DEFAULT_TEMPLATES["interrupt"]

    def test_empty_string_override_is_returned_as_is(self):
        """A deliberately empty template suppresses that mode -- the
        rendering layer detects empty and short-circuits the ack send."""
        assert resolve_busy_ack_template({"queue": ""}, "queue") == ""

    def test_whitespace_only_override_is_returned_as_is(self):
        """Whitespace-only is also treated as "suppress this mode"."""
        assert resolve_busy_ack_template({"queue": "   "}, "queue") == "   "


# ---------------------------------------------------------------------------
# render_busy_ack
# ---------------------------------------------------------------------------


class TestRenderBusyAck:
    def test_render_default_with_status(self):
        rendered = render_busy_ack(None, "interrupt", status_detail=" (1 min elapsed)")
        assert rendered == (
            "⚡ Interrupting current task (1 min elapsed). "
            "I'll respond to your message shortly."
        )

    def test_render_default_without_status(self):
        rendered = render_busy_ack(None, "queue", status_detail="")
        assert "Queued for the next turn." in rendered
        # No extra space from an empty status detail
        assert "  " not in rendered.replace(". ", "")

    def test_render_custom_template_with_placeholder(self):
        templates = {
            "queue": "Got it{status_detail}. Will reply when the current task wraps."
        }
        rendered = render_busy_ack(templates, "queue", status_detail=" (3 min)")
        assert rendered == "Got it (3 min). Will reply when the current task wraps."

    def test_render_template_without_placeholder(self):
        """Operators may omit the placeholder -- the status string is
        computed but discarded; the template is rendered as-is."""
        templates = {"interrupt": "Working on it, hold."}
        rendered = render_busy_ack(templates, "interrupt", status_detail=" (long elapsed)")
        assert rendered == "Working on it, hold."

    def test_render_unknown_placeholder_falls_back_to_default(self):
        """A typo'd placeholder in a user template must not crash the
        busy-turn path. We render the default instead so the user still
        sees feedback."""
        templates = {"queue": "Hold on{bogus_placeholder} please."}
        rendered = render_busy_ack(templates, "queue", status_detail=" (10s)")
        assert "Queued for the next turn (10s)" in rendered

    def test_render_empty_template_returns_empty(self):
        assert render_busy_ack({"queue": ""}, "queue", status_detail=" (x)") == ""

    def test_render_japanese_template(self):
        """The example in the issue is a Japanese persona override.
        Verify Unicode passes through cleanly."""
        templates = {
            "queue": "承知しました。{status_detail}現在の処理が終わり次第お返事します。"
        }
        rendered = render_busy_ack(templates, "queue", status_detail="（5分経過）")
        assert "承知しました" in rendered
        assert "（5分経過）" in rendered


# ---------------------------------------------------------------------------
# Env-var roundtrip
# ---------------------------------------------------------------------------


class TestEnvVarRoundtrip:
    def test_encode_decode_full_roundtrip(self):
        templates = {
            "queue": "Q {status_detail}",
            "interrupt": "I {status_detail}",
            "steer": "S {status_detail}",
        }
        encoded = encode_templates_for_env(templates)
        assert encoded is not None
        decoded = load_templates_from_env({ENV_VAR_TEMPLATES: encoded})
        assert decoded == templates

    def test_encode_drops_non_string_values(self):
        """Defensive: a YAML override with a number or list must not
        survive the env transport (subprocess would crash on parse)."""
        templates = {"queue": "ok", "interrupt": 42, "steer": ["a"]}
        encoded = encode_templates_for_env(templates)
        assert encoded is not None
        decoded = json.loads(encoded)
        assert decoded == {"queue": "ok"}

    def test_encode_drops_unknown_modes(self):
        templates = {"queue": "ok", "BOGUS_MODE": "should not survive"}
        encoded = encode_templates_for_env(templates)
        decoded = json.loads(encoded)
        assert "BOGUS_MODE" not in decoded
        assert decoded["queue"] == "ok"

    def test_encode_empty_returns_none(self):
        """Nothing to bridge → return None so the caller skips setting
        the env var entirely (cleaner than setting it to ``{}``)."""
        assert encode_templates_for_env(None) is None
        assert encode_templates_for_env({}) is None
        assert encode_templates_for_env({"queue": 42}) is None  # all dropped
        assert encode_templates_for_env("not-a-mapping") is None

    def test_load_missing_env_var_returns_empty(self):
        assert load_templates_from_env({}) == {}

    def test_load_malformed_json_returns_empty(self):
        assert load_templates_from_env({ENV_VAR_TEMPLATES: "{not json"}) == {}

    def test_load_non_object_json_returns_empty(self):
        assert load_templates_from_env({ENV_VAR_TEMPLATES: '["queue"]'}) == {}
        assert load_templates_from_env({ENV_VAR_TEMPLATES: '"queue"'}) == {}

    def test_load_drops_non_string_values_on_decode(self):
        """If somehow a non-string value reaches the env (process
        manipulating env directly), drop it on decode too."""
        raw = json.dumps({"queue": "ok", "interrupt": 42, "steer": None})
        loaded = load_templates_from_env({ENV_VAR_TEMPLATES: raw})
        assert loaded == {"queue": "ok"}

    def test_load_drops_unknown_modes_on_decode(self):
        raw = json.dumps({"queue": "ok", "BOGUS": "no", "steer": "yes"})
        loaded = load_templates_from_env({ENV_VAR_TEMPLATES: raw})
        assert loaded == {"queue": "ok", "steer": "yes"}
