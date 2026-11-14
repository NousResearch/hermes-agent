"""Direct FAL requests are signed with the serving profile's own ``FAL_KEY``.

The module-level ``fal_client.submit`` is one process-wide ``SyncClient()`` that reads
``os.environ`` (the launch profile's ``.env``) on first use and keeps that key, so on a
multiplexed gateway a routed profile's image or video job was billed to the launch profile.
Each direct submit must build its client from the active profile scope instead.
"""

import types

import pytest


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Launch profile A (its key exported into the process env, as the gateway start does) and a
    routed profile B with its own key; multiplexing on. Restores the process-wide scope state."""
    from agent import secret_scope
    from tools import fal_common

    root = tmp_path / ".hermes"
    homes = {"A": root, "B": root / "profiles" / "b"}
    keys = {"A": "fal-key-launch-A", "B": "fal-key-routed-B"}
    for label, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / ".env").write_text(f"FAL_KEY={keys[label]}\n")
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("FAL_KEY", keys["A"])
    monkeypatch.setattr(fal_common, "_direct_clients", {}, raising=False)
    was_active = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        yield homes, {value: label for label, value in keys.items()}
    finally:
        secret_scope.set_multiplex_active(was_active)


def _recording_fal_client(label_of):
    """A ``fal_client`` stand-in: ``SyncClient(key=)`` records the key and its ``submit`` records
    which key signed the job; the module-level ``submit`` records the process-env key it would use."""
    import os

    constructed, signed = [], []

    def _client(key=None):
        constructed.append(label_of.get(key, key))
        return types.SimpleNamespace(submit=lambda *a, **k: signed.append(label_of.get(key, key)) or "handle")

    module = types.SimpleNamespace(
        SyncClient=_client,
        submit=lambda *a, **k: signed.append(label_of.get(os.environ.get("FAL_KEY"))) or "handle")
    return module, constructed, signed


def _submit_image(module, monkeypatch):
    from tools import image_generation_tool as image_tool
    monkeypatch.setattr(image_tool, "fal_client", module)
    return image_tool._submit_fal_request("fal-ai/flux-2/klein/9b", {"prompt": "x"})


def _submit_video(module, monkeypatch):
    from plugins.video_gen import fal as fal_plugin
    monkeypatch.setattr(fal_plugin, "_fal_client", module)
    return fal_plugin._submit_fal_video_request("fal-ai/veo3.1", {"prompt": "x"})


@pytest.mark.parametrize("submit", [_submit_image, _submit_video], ids=["image", "video"])
def test_routed_turns_sign_with_their_own_profile_key(two_profiles, monkeypatch, submit):
    from agent.runtime_scope import bind_profile_scope

    homes, label_of = two_profiles
    module, constructed, signed = _recording_fal_client(label_of)
    for label in ("B", "A", "B"):
        with bind_profile_scope(homes[label], hydrate_secrets=False):
            assert submit(module, monkeypatch) == "handle"

    assert signed == ["B", "A", "B"]
    # One client per key, reused across turns rather than rebuilt (and leaked) per request.
    assert constructed == ["B", "A"]


@pytest.mark.parametrize("submit", [_submit_image, _submit_video], ids=["image", "video"])
def test_single_profile_run_keeps_its_env_key(tmp_path, monkeypatch, submit):
    from tools import fal_common

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("FAL_KEY", "fal-key-shell")
    monkeypatch.setattr(fal_common, "_direct_clients", {}, raising=False)
    module, constructed, signed = _recording_fal_client({"fal-key-shell": "shell"})

    assert submit(module, monkeypatch) == "handle"
    assert constructed == ["shell"] and signed == ["shell"]
