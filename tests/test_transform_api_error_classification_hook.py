"""Tests for the ``transform_api_error_classification`` plugin hook.

Covers the seam in ``agent.error_classifier.classify_api_error`` (step 0,
consulted before the built-in pipeline) and the sanitization contract of
``hermes_cli.plugins.get_plugin_error_classification``.

The fixture error is deliberately synthetic (fake provider, made-up
message, no status code) so no present or future built-in rule can claim
it — the earlier OpenRouter tool-use-404 fixture went stale the moment
core learned that exact phrase.

Mirrors the ``transform_tool_result`` hook tests: patch the symbol the
call site actually imports (``hermes_cli.plugins.*``) rather than the
consuming module, because the import happens at call time.
"""

import importlib.util
import logging

import hermes_cli.plugins as plugins_mod
from agent.error_classifier import FailoverReason, classify_api_error


class _FakeAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.body = body or {}


_UNCLAIMED_MESSAGE = "flux capacitor drift detected in shard seven"


def _classify_unclaimed_error(**kwargs):
    return classify_api_error(
        # No status code on purpose: the built-in pipeline claims whole
        # status classes (4xx -> format_error, 429 -> rate_limit, ...), so
        # a status-less neutral message is the only shape guaranteed to
        # reach the unknown/retryable fall-through.
        _FakeAPIError(_UNCLAIMED_MESSAGE),
        provider="acmecloud",
        model="acme/large-1",
        **kwargs,
    )


# ── Baseline: no plugins ────────────────────────────────────────────────


def test_no_hook_falls_through_to_builtin(monkeypatch):
    # Fresh manager so no stale plugin hooks pollute state.
    monkeypatch.setattr(plugins_mod, "_plugin_manager", plugins_mod.PluginManager())

    result = _classify_unclaimed_error()
    # The synthetic error matches no built-in rule: unknown/retryable is
    # the pipeline's fall-through, which is exactly the class of error
    # this hook lets provider plugins claim.
    assert result.reason == FailoverReason.unknown
    assert result.retryable is True


# ── Plugin classification wins over built-ins ───────────────────────────


def test_plugin_classification_wins(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [
            {"reason": "model_not_found", "retryable": False, "should_fallback": True}
        ],
    )

    result = _classify_unclaimed_error()
    assert result.reason == FailoverReason.model_not_found
    assert result.retryable is False
    assert result.should_fallback is True
    # Extracted context is preserved on the ClassifiedError.
    assert result.provider == "acmecloud"
    assert result.status_code is None

# ── Invalid returns are ignored, first valid wins ───────────────────────


def test_invalid_reason_falls_through_to_builtin(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{"reason": "not_a_real_reason"}],
    )

    result = _classify_unclaimed_error()
    assert result.reason == FailoverReason.unknown

def test_first_valid_result_wins(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [
            {"reason": "bogus"},
            {"reason": "billing"},
            {"reason": "rate_limit"},
        ],
    )

    result = _classify_unclaimed_error()
    assert result.reason == FailoverReason.billing


def test_skipped_valid_results_log_runtime_warning(monkeypatch, caplog):
    # The #64714 skipped-transform rule: a valid-but-losing classification
    # must surface in logs, never be silently shadowed. Invalid results
    # (here "bogus") are not "skipped valid" and must not count.
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [
            {"reason": "bogus"},
            {"reason": "billing"},
            {"reason": "rate_limit"},
        ],
    )

    with caplog.at_level(logging.WARNING, logger=plugins_mod.logger.name):
        result = _classify_unclaimed_error()
    assert result.reason == FailoverReason.billing
    warnings = [r.getMessage() for r in caplog.records if "skipped" in r.getMessage()]
    assert len(warnings) == 1
    assert "skipped 1 valid" in warnings[0]

    # A lone winner is not a conflict: no warning.
    caplog.clear()
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{"reason": "billing"}],
    )
    with caplog.at_level(logging.WARNING, logger=plugins_mod.logger.name):
        result = _classify_unclaimed_error()
    assert result.reason == FailoverReason.billing
    assert not [r for r in caplog.records if "skipped" in r.getMessage()]


def test_helper_exception_never_breaks_classification(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("plugin infrastructure exploded")

    monkeypatch.setattr(plugins_mod, "get_plugin_error_classification", _boom)

    result = _classify_unclaimed_error()
    assert result.reason == FailoverReason.unknown
    assert result.retryable is True


# ── Hook kwargs contract ────────────────────────────────────────────────

def test_message_override_and_error_context_sanitized(monkeypatch):
    monkeypatch.setattr(
        plugins_mod, "invoke_hook",
        lambda name, **kw: [{
            "reason": "model_not_found",
            "message": "  custom guidance  ",
            "error_context": {"upstream_provider": "AcmeCloud"},
        }],
    )

    result = _classify_unclaimed_error()
    assert result.message == "custom guidance"
    assert result.error_context == {"upstream_provider": "AcmeCloud"}


# ── Plugin register() end-to-end (synthetic, written at test time) ──────

_SYNTHETIC_PLUGIN = '''
def classify(provider=None, error_message=None, **kwargs):
    """Self-scoped classifier for acmecloud's flux-drift errors."""
    if provider != "acmecloud":
        return None
    if "flux capacitor drift" not in (error_message or ""):
        return None
    return {"reason": "overloaded", "retryable": True, "should_fallback": True}


def register(ctx):
    ctx.register_hook("transform_api_error_classification", classify)
'''


def _load_synthetic_plugin(tmp_path):
    plugin_init = tmp_path / "acmecloud_classifier.py"
    plugin_init.write_text(_SYNTHETIC_PLUGIN, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("acmecloud_classifier", plugin_init)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_synthetic_plugin_end_to_end(tmp_path, monkeypatch):
    """register() + real invoke_hook + classify_api_error, no mocks."""
    demo = _load_synthetic_plugin(tmp_path)
    manager = plugins_mod.PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)

    class _Ctx:
        def register_hook(self, name, cb):
            manager._hooks.setdefault(name, []).append(cb)

    demo.register(_Ctx())

    result = _classify_unclaimed_error()
    assert result.reason == FailoverReason.overloaded
    assert result.retryable is True
    assert result.should_fallback is True

    # And the built-in pipeline is untouched for everything the plugin
    # doesn't claim.
    other = classify_api_error(
        _FakeAPIError("rate limit exceeded", status_code=429),
        provider="acmecloud",
    )
    assert other.reason == FailoverReason.rate_limit
