"""Evidence rules for the config-level "SUBAGENT MODEL REJECTED" delegation notice.

Only a FAILED result's ``error`` (or its structured ``failure_reason``) is evidence; prose
summaries and completed results never are, and a route that served another model is not the
configured default's fault. Every rejection phrase is bound to the model it is about (a phrase
naming another model never accuses the configured one); structured evidence needs a served model
equal to the configured one; model identity is the FULL id (only the aggregator transport prefix
may be dropped); and an ordinary failed sibling is counted as "failed for another reason", never
as "no recorded outcome".
"""

import pytest

import tools.process_registry_notifications as _prn
from tools.process_registry_notifications import format_process_notification

MODEL = "upstage/solar-pro-4"
REJECTION = f"HTTP 400: {MODEL} is not a valid model ID"
BANNER = "SUBAGENT MODEL REJECTED"
PROSE = f"Audited {MODEL} usage; the authorization file does not exist, so I skipped it."


def _cfg(monkeypatch, **over):
    cfg = {"model": MODEL, "provider": "openrouter", **over}
    monkeypatch.setattr(_prn, "_delegation_config", lambda: cfg)
    return cfg


def _render(results):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_rej",
        "is_batch": True,
        "results": results,
        "goals": [r.get("goal", "") for r in results],
        "session_key": "agent:main:cli:dm:local",
        "status": "completed",
    }
    text = format_process_notification(evt)
    assert text is not None
    return text


def _failed(**over):
    return {"task_index": 0, "status": "failed", "goal": "A", "error": REJECTION, **over}


@pytest.mark.parametrize("status", ["completed", "success"])
def test_completed_summary_mentioning_model_and_missing_file_is_not_a_rejection(monkeypatch, status):
    cfg = _cfg(monkeypatch)
    results = [{"task_index": 0, "status": status, "goal": "A", "summary": PROSE}]
    assert _prn._delegation_model_not_found(results, cfg) is False
    assert _prn._delegation_model_not_found_notice(results) is None
    text = _render(results)
    assert BANNER not in text
    assert PROSE in text


def test_genuine_failed_model_error_renders_actionable_notice(monkeypatch):
    cfg = _cfg(monkeypatch)
    results = [_failed(summary="irrelevant prose")]
    assert _prn._delegation_model_not_found(results, cfg) is True
    text = _render(results)
    assert text.count(BANNER) == 1
    assert f'"{MODEL}"' in text and 'provider "openrouter"' in text
    assert f"Evidence (task 1): {REJECTION}" in text
    assert "HTTP 400: not a valid model ID" not in text
    assert "before doing any work" not in text
    assert "All 1 tasks in this batch failed with this model rejection." in text
    assert "hermes config get delegation.model" in text
    assert "No fallback chain is configured (delegation.fallback_providers)." in text


def test_failed_non_provider_file_not_found_is_not_a_rejection(monkeypatch):
    cfg = _cfg(monkeypatch)
    for error in (
        "the authorization file /root/.hermes/auth.json does not exist",
        f"the authorization file for {MODEL} does not exist",
    ):
        results = [_failed(error=error)]
        assert _prn._delegation_model_not_found(results, cfg) is False
        assert BANNER not in _render(results)


@pytest.mark.parametrize("error", [f"model {MODEL} does not exist", f"{MODEL} does not exist", f"model '{MODEL}' does not exist"])
def test_generic_does_not_exist_counts_when_model_is_the_subject(monkeypatch, error):
    cfg = _cfg(monkeypatch)
    assert _prn._delegation_model_not_found([_failed(error=error)], cfg) is True


def test_mixed_batch_does_not_claim_every_task_failed(monkeypatch):
    _cfg(monkeypatch)
    results = [
        _failed(),
        {"task_index": 1, "status": "completed", "goal": "B", "summary": "B finished: wrote the bridge module"},
    ]
    text = _render(results)
    assert text.count(BANNER) == 1
    assert "Every task in this batch failed" not in text
    assert "before doing any work" not in text
    assert "HTTP 400: not a valid model ID" not in text
    assert "1 of 2 tasks in this batch failed with this model rejection; 1 succeeded." in text
    assert "B finished: wrote the bridge module" in text


def test_named_route_serving_another_model_is_not_blamed_on_the_default(monkeypatch):
    cfg = _cfg(monkeypatch)
    error = "HTTP 404: other/model-x does not exist"
    for served in ({"model": "other/model-x"}, {"route": {"model": "other/model-x", "provider": "acme"}}):
        results = [_failed(error=error, **served)]
        assert _prn._delegation_model_not_found(results, cfg) is False
        text = _render(results)
        assert BANNER not in text
        assert error in text  # still visible in the per-task wall


def test_route_serving_the_configured_model_under_a_prefix_is_config_level(monkeypatch):
    cfg = _cfg(monkeypatch)
    results = [_failed(error="model_not_found", route={"model": f"openrouter/{MODEL}", "provider": "acme"})]
    assert _prn._delegation_model_not_found(results, cfg) is True
    assert 'provider "acme"' in "\n".join(_prn._delegation_model_not_found_notice(results))


@pytest.mark.parametrize("status", [None, "", "unknown"])
def test_unknown_status_is_not_a_confirmed_failure(monkeypatch, status):
    cfg = _cfg(monkeypatch)
    result = {"task_index": 0, "goal": "A", "error": REJECTION}
    if status is not None:
        result["status"] = status
    assert _prn._delegation_model_not_found([result], cfg) is False
    text = _render([result])
    assert BANNER not in text
    assert REJECTION in text

    structured = {**result, "failure_reason": "model_not_found"}
    assert _prn._delegation_model_not_found([structured], cfg) is True
    assert BANNER in _render([structured])


def test_no_configured_model_means_no_config_level_rejection(monkeypatch):
    assert _prn._delegation_model_not_found([_failed()], {"provider": "openrouter"}) is False


def test_outcome_counts_with_succeeded_and_unrecorded_tasks(monkeypatch):
    _cfg(monkeypatch)
    results = [
        _failed(),
        {"task_index": 1, "status": "completed", "goal": "B", "summary": "ok"},
        {"task_index": 2, "goal": "C"},
    ]
    notice = "\n".join(_prn._delegation_model_not_found_notice(results))
    assert "1 of 3 tasks in this batch failed with this model rejection; 1 succeeded; 1 have no recorded outcome." in notice


def test_fallback_chain_line_is_absent_when_a_chain_is_configured(monkeypatch):
    _cfg(monkeypatch, fallback_providers=[{"provider": "openrouter", "model": "other/model-x"}])
    assert "No fallback chain is configured" not in _render([_failed()])


def test_failed_error_naming_a_different_model_is_not_blamed_on_the_configured_model(monkeypatch):
    cfg = _cfg(monkeypatch, model="glm-5.3-flash")
    error = "model different-model is not a valid model ID"
    results = [{"task_index": 0, "status": "failed", "model": "glm-5.3-flash", "error": error}]
    assert _prn._delegation_model_not_found(results, cfg) is False
    text = _render(results)
    assert BANNER not in text
    assert error in text


def test_route_serving_a_different_namespace_model_is_not_blamed_on_the_configured_model(monkeypatch):
    cfg = _cfg(monkeypatch)
    error = "model other/solar-pro-4 is not a valid model ID"
    results = [{"task_index": 0, "status": "failed", "route": {"model": "other/solar-pro-4"}, "error": error}]
    assert _prn._delegation_model_not_found(results, cfg) is False
    assert BANNER not in _render(results)


def test_ordinary_failed_sibling_is_counted_apart_from_unknown_status(monkeypatch):
    _cfg(monkeypatch)
    results = [_failed(), {"task_index": 1, "status": "failed", "error": "FileNotFoundError: /root/.hermes/auth.json"}]
    text = _render(results)
    assert "1 of 2 tasks in this batch failed with this model rejection; 1 failed for another reason." in text
    assert "no recorded outcome" not in text
    assert "Every task in this batch failed" not in text


def test_structured_model_not_found_without_model_binding_is_not_a_config_rejection(monkeypatch):
    cfg = _cfg(monkeypatch)
    results = [{"task_index": 0, "status": "failed", "error": "model_not_found", "failure_reason": "model_not_found"}]
    assert _prn._delegation_model_not_found(results, cfg) is False
    assert BANNER not in _render(results)


def test_transport_prefix_alias_still_binds_to_the_configured_model(monkeypatch):
    cfg = _cfg(monkeypatch)
    results = [{"task_index": 0, "status": "failed", "error": "model_not_found",
                "route": {"model": f"openrouter/{MODEL}", "provider": "acme"}}]
    assert _prn._delegation_model_not_found(results, cfg) is True
    assert 'provider "acme"' in "\n".join(_prn._delegation_model_not_found_notice(results))


def test_same_model_preserves_namespace_identity():
    assert _prn._same_model("other/solar-pro-4", "upstage/solar-pro-4") is False
    assert _prn._same_model("upstage/solar-pro-4", "upstage/solar-pro-4") is True
    assert _prn._same_model("openrouter/upstage/solar-pro-4", "upstage/solar-pro-4") is True
    assert _prn._same_model("openrouter/solar-pro-4", "solar-pro-4") is False
