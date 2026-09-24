from pathlib import Path

import tui_gateway.server as server


def test_model_options_binds_requested_profile_home(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "fred-work"
    profile_home.mkdir(parents=True)
    seen = {}

    monkeypatch.setattr(
        server,
        "_profile_home",
        lambda name: profile_home if name == "fred-work" else None,
    )
    monkeypatch.setattr(server, "_model_picker_context", lambda agent: object())

    def build_payload(ctx, **kwargs):
        from hermes_constants import get_hermes_home

        seen["home"] = Path(get_hermes_home())
        return {"providers": []}

    monkeypatch.setattr("hermes_cli.inventory.build_model_options_payload", build_payload)

    response = server._methods["model.options"](1, {"profile": "fred-work"})

    assert response["result"] == {"providers": []}
    assert seen["home"] == profile_home