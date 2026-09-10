"""The gateway's config read must honour profile inheritance.

A profile that opts in with `inherit: true` and declares no model of its own
resolves through `hermes_cli.config.load_config` to the root model. The gateway
reads config by a different path — `_load_cfg` — which layered the managed
overlay and env expansion but not inheritance. An inheriting profile therefore
looked model-less here, and `_resolve_model` fell through to the cost-safe
silent default (`z-ai/glm-5.2`), which the configured Anthropic endpoint
rejects with a 404. The room showed "thinking..." forever while the retries
failed out of sight.

Same bug family as the one fixed in cli.py's `load_cli_config`: several readers
of one file, only some of them applying the rules.
"""

import textwrap
import threading
from pathlib import Path

import pytest


@pytest.fixture
def inheriting_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A root config with a model, and a profile that inherits it.

    The gateway caches its config read in module globals keyed by path, so the
    fixture points the module at the temp profile and clears that cache.
    monkeypatch restores both, which keeps this file from leaking a fake
    HERMES_HOME into every test that runs after it.
    """
    root = tmp_path / "hermes"
    root.mkdir()
    (root / "config.yaml").write_text(
        textwrap.dedent(
            """\
            model:
              provider: anthropic
              base_url: https://api.anthropic.com
              default: claude-opus-5
            """
        )
    )

    profile = root / "profiles" / "notion-expert"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        textwrap.dedent(
            """\
            inherit: true
            agent:
              max_turns: 45
            """
        )
    )

    from tui_gateway import server

    monkeypatch.setattr(server, "_hermes_home", str(profile))
    monkeypatch.setattr(server, "_cfg_cache", None, raising=False)
    monkeypatch.setattr(server, "_cfg_mtime", None, raising=False)
    monkeypatch.setattr(server, "_cfg_path", None, raising=False)
    return profile


def test_gateway_config_read_resolves_an_inherited_model(
    inheriting_profile: Path,
) -> None:
    """`_load_cfg` must see the inherited model, not an empty one."""
    from tui_gateway import server

    cfg = server._load_cfg()

    assert cfg.get("model", {}).get("default") == "claude-opus-5"


def test_the_profiles_own_keys_still_win_over_the_inherited_root(
    inheriting_profile: Path,
) -> None:
    """Inheritance fills gaps; it must not overwrite what the profile declares."""
    from tui_gateway import server

    cfg = server._load_cfg()

    assert cfg.get("agent", {}).get("max_turns") == 45


def test_inheriting_profile_does_not_fall_back_to_the_silent_default(
    inheriting_profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 404 the user actually saw: an inherited model must not become glm."""
    from tui_gateway import server

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)

    assert server._resolve_model() == "claude-opus-5"


def test_concurrent_profile_reads_keep_their_own_inherited_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent cache update must not redirect another profile's inheritance."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server

    profiles: dict[str, Path] = {}
    for name in ("A", "B"):
        root = tmp_path / name
        profile = root / "profiles" / "bot"
        profile.mkdir(parents=True)
        (root / "config.yaml").write_text(f"model:\n  default: model-{name}\n")
        (profile / "config.yaml").write_text("inherit: true\n")
        profiles[name] = profile

    monkeypatch.setattr(server, "_cfg_cache", None, raising=False)
    monkeypatch.setattr(server, "_cfg_mtime", None, raising=False)
    monkeypatch.setattr(server, "_cfg_path", None, raising=False)

    barrier = threading.Barrier(2)
    apply_managed = server._apply_managed

    def synchronize_after_raw_read(cfg: dict) -> dict:
        barrier.wait(timeout=5)
        return apply_managed(cfg)

    monkeypatch.setattr(server, "_apply_managed", synchronize_after_raw_read)
    results: dict[str, str] = {}

    def load(name: str) -> None:
        token = set_hermes_home_override(profiles[name])
        try:
            results[name] = server._load_cfg()["model"]["default"]
        finally:
            reset_hermes_home_override(token)

    threads = [threading.Thread(target=load, args=(name,)) for name in profiles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert results == {"A": "model-A", "B": "model-B"}
