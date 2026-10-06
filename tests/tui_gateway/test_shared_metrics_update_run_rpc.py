"""shared_metrics.update_run: Desktop reports a packaged self-update as bounded kind=desktop dims only."""

from __future__ import annotations

from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
import tui_gateway.server as server


def test_desktop_update_run_is_bounded_and_never_carries_raw_text(monkeypatch):
    rows: list[tuple[str, dict]] = []
    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: True)
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", lambda mark, data: rows.append((mark, data)))

    response = server.handle_request({"jsonrpc": "2.0", "id": "r1", "method": "shared_metrics.update_run", "params": {
        "outcome": "failed", "failed_stage": "ENOENT: /Users/alice/Library/Caches/update.zip",
        "mechanism": "electron-updater", "duration_ms": 400_000, "from_commit_date": 0,
    }})

    assert response["result"] == {"ok": True}
    assert rows == [(contract.UPDATE_RUN_MARK, {
        "apply_mode": "package", "duration_bucket": "5m_to_15m", "failed_stage": "other",
        "from_version_age_bucket": "gte_90d", "kind": "desktop", "outcome": "failed",
    })]
    assert contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, rows[0][1])
