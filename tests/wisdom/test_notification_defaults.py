"""Exercise notification defaults through real profile configuration reads/writes."""

import pytest
from unittest.mock import Mock

from hermes_cli.config import load_config, read_raw_config, save_config
from hermes_wisdom.agent_led.policy import load_policy
from hermes_wisdom.mediation import delivery_mode
from hermes_wisdom.consumption import WisdomConsumption
from hermes_wisdom.store import WisdomStore


def test_default_advice_mode_survives_config_save_without_enabling_wisdom(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    assert delivery_mode() == "agent"
    assert delivery_mode(config["wisdom"]) == "agent"
    assert config["wisdom"]["enabled"] is False
    save_config(config)
    assert "delivery_mode" not in read_raw_config().get("wisdom", {}).get("notifications", {})
    assert delivery_mode() == "agent"
    assert load_config()["wisdom"]["enabled"] is False
    manager = WisdomConsumption(
        store=WisdomStore(tmp_path / "wisdom"), client=Mock(), scan=Mock(),
        config=load_config()["wisdom"],
    )
    monkeypatch.setattr(manager.store, "feed_events", lambda **_kwargs: pytest.fail(
        "Default agent mode must not also consume the fixed transport queue"
    ))
    assert manager.dispatch_telegram() == {"attempted": False, "delivered": 0}
    assert manager.dispatch_slack() == {"attempted": False, "delivered": 0}


@pytest.mark.parametrize("mode", ["fixed", "agent"])
def test_saved_delivery_choice_survives_reload_and_fixed_skips_agent_policy(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = load_config()
    config["wisdom"]["notifications"]["delivery_mode"] = mode
    save_config(config, preserve_keys={("wisdom", "notifications", "delivery_mode")})
    assert delivery_mode() == mode
    save_config(load_config())
    assert read_raw_config()["wisdom"]["notifications"]["delivery_mode"] == mode
    assert delivery_mode() == mode
    if mode == "fixed":
        class NoAgentPolicy:
            def agent_led_policy(self):
                pytest.fail("Fixed-copy opt-out must not start agent policy work")

        assert load_policy(client=NoAgentPolicy()).enabled is False
