"""cost_budget_status — the agent's read-only view of its own spend and budget."""
import json

import pytest

import tools.cost_budget_tool as cbt
from agent import cost_budget


@pytest.fixture(autouse=True)
def _reset():
    cost_budget._reset_for_tests()
    yield
    cost_budget._reset_for_tests()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _spend(monkeypatch, *, day=0.0, month=0.0, unpriced=0, runs=(), upper=False):
    def fake(home, now):
        return cost_budget.SpendSnapshot(
            day_key=now.strftime("%Y-%m-%d"), month_key=now.strftime("%Y-%m"), day_usd=day,
            month_usd=month, day_unpriced_tokens=unpriced, month_unpriced_tokens=unpriced,
            unpriced_models=("m1",) if unpriced else (), runs=tuple(runs), as_of=now.timestamp(),
            day_upper_bound=upper)

    monkeypatch.setattr(cost_budget, "compute_spend", fake)


def _run():
    return json.loads(cbt._cost_budget_status_handler({}))


def _run_spend(job_id, cost, root):
    return cost_budget.RunSpend(root_id=root, job_id=job_id, started_at=1.0,
                                cost_usd=cost, unpriced_tokens=0)


def test_reports_spend_and_remaining(home, monkeypatch):
    (home / "config.yaml").write_text(
        "cost_budget:\n  daily_usd: 10\n  monthly_usd: 100\n  per_run_usd: 2\n"
        "  discretionary_jobs: [explore]\n", encoding="utf-8")
    runs = [_run_spend("a", 0.5, "r1"), _run_spend("b", 3.0, "r2")]
    _spend(monkeypatch, day=4, month=120, runs=runs)

    out = _run()

    assert out["configured"] is True
    assert out["today"] == {"spent_usd": 4.0, "budget_usd": 10.0, "remaining_usd": 6.0, "fraction": 0.4}
    assert out["month"]["remaining_usd"] == 0.0
    assert out["over_budget"] is True
    assert out["discretionary_jobs_skipped"] is True
    assert out["discretionary_jobs"] == ["explore"]
    assert out["per_run_usd"] == 2.0
    assert out["spend_is_lower_bound"] is False
    assert out["spend_is_upper_bound"] is False
    assert out["config_warnings"] == []
    assert [r["job_id"] for r in out["costliest_runs_this_month"]] == ["b", "a"]


def test_unconfigured_reads_nothing(home, monkeypatch):
    monkeypatch.setattr(cost_budget, "compute_spend", lambda *a, **k: pytest.fail("scanned"))

    out = _run()

    assert out["configured"] is False
    assert "not tracked" in out["message"]
    assert not (home / "cost_budget").exists()


def test_lower_bound_flag_with_unpriced(home, monkeypatch):
    (home / "config.yaml").write_text("cost_budget:\n  daily_usd: 10\n", encoding="utf-8")
    _spend(monkeypatch, day=1, month=1, unpriced=5000)

    out = _run()

    assert out["spend_is_lower_bound"] is True
    assert out["today"]["remaining_usd"] is None
    assert "no price" in out["today"]["remaining_note"]
    assert out["unpriced_tokens_today"] == 5000
    assert out["unpriced_models"] == ["m1"]


def test_store_error_is_a_tool_error(home, monkeypatch):
    (home / "config.yaml").write_text("cost_budget:\n  daily_usd: 10\n", encoding="utf-8")
    def boom(*a, **k):
        raise RuntimeError("locked")

    monkeypatch.setattr(cost_budget, "compute_spend", boom)
    assert "error" in _run()


def test_registered_in_own_toolset():
    from tools.registry import registry
    from toolsets import TOOLSETS

    assert registry.get_toolset_for_tool("cost_budget_status") == "cost_budget"
    assert TOOLSETS["cost_budget"]["tools"] == ["cost_budget_status"]


def test_upper_bound_and_config_warnings_surfaced(home, monkeypatch):
    (home / "config.yaml").write_text("cost_budget:\n  daily_usd: 10\n  alert_at: [80]\n",
                                      encoding="utf-8")
    _spend(monkeypatch, day=3, month=3, upper=True)

    out = _run()

    assert out["spend_is_upper_bound"] is True
    assert out["today"]["remaining_usd"] == 7.0
    assert any("fraction" in w for w in out["config_warnings"])
