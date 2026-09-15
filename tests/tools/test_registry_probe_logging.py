"""Unavailable probes are routine; grace must re-probe and exceptions stay visible."""

import importlib
import logging
from types import SimpleNamespace

import pytest

registry_module = importlib.import_module("tools.registry")


@pytest.mark.parametrize("raises, level", [(False, logging.INFO), (True, logging.WARNING)])
def test_grace_failure_reprobes_with_safe_callable_label(monkeypatch, caplog, raises, level):
    clock = [1000.0]
    monkeypatch.setattr(registry_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(registry_module, "_check_fn_cache", {})
    monkeypatch.setattr(registry_module, "_check_fn_last_good", {})
    monkeypatch.setattr(registry_module, "check_fn_cache_scope", lambda: "test-profile")
    calls = []

    class Probe:
        def __repr__(self):
            raise AssertionError("callable repr must never be logged")

        def __call__(self):
            calls.append(None)
            if len(calls) == 1:
                return True
            if raises:
                raise RuntimeError("probe failed")
            return False

    probe = Probe()
    reg = registry_module.ToolRegistry()
    reg.register(name="test-probe", toolset="test-probes", schema={}, handler=lambda args: "{}", check_fn=probe)
    with caplog.at_level(logging.INFO, logger="tools.registry"):
        assert reg.is_toolset_available("test-probes") is True
        clock[0] += registry_module._CHECK_FN_TTL_SECONDS + 1
        assert reg.is_toolset_available("test-probes") is True
        assert reg.is_toolset_available("test-probes") is True
        assert len(calls) == 3  # a grace failure must never be cached
        clock[0] += registry_module._CHECK_FN_FAILURE_GRACE_SECONDS + 1
        assert reg.is_toolset_available("test-probes") is False
    records = [r for r in caplog.records if r.name == "tools.registry"]
    assert len(records) == 3
    assert all(record.levelno == level for record in records)
    if raises:
        assert records[-1].exc_info is not None
        assert isinstance(records[-1].exc_info[1], RuntimeError)
        assert str(records[-1].exc_info[1]) == "probe failed"
    else:
        assert records[-1].exc_info is None
