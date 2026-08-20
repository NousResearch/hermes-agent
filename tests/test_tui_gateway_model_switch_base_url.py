"""``tui_gateway.server._persist_model_switch`` × ``custom_providers``.

Companion to ``tests/hermes_cli/test_web_server_picker_base_url.py`` — the
dashboard picker and the chat TUI's ``/model`` slash command both persist
``model.base_url`` and share one bug class: blanking the URL when switching to a
provider whose endpoint only exists in the user's config strands routing.

The important detail here is the SPELLING of the target. ``switch_model()``
reports custom providers by canonical slug — ``custom:<normalized-name>``, see
``hermes_cli.providers.custom_provider_slug`` — never by the raw
``custom_providers[].name``, no matter which form the caller passed in. Any fix
that compares ``result.target_provider`` against raw names therefore never fires
on a real switch, so every case below uses the slug the resolver actually emits.
"""

from types import SimpleNamespace

import pytest
import yaml

LITELLM_URL = "http://192.168.1.10:4000"
VLLM_URL = "http://192.168.1.11:8000"

CONFIG = f"""\
custom_providers:
- name: my-litellm
  base_url: {LITELLM_URL}
  api_key: sk-test-litellm
  api_format: openai
- name: my-vllm
  base_url: {VLLM_URL}
  api_key: sk-test-vllm
  api_format: openai

model:
  default: model-a
  provider: custom:my-litellm
  base_url: {LITELLM_URL}
  api_key: sk-test-litellm
"""


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME holding the two-custom-provider config."""
    home = tmp_path / "tui_home"
    home.mkdir()
    (home / ".env").write_text("")
    (home / "config.yaml").write_text(CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _persist(target_provider, base_url=None, new_model="model-b"):
    """Run the real persist path with a switch result shaped like switch_model's."""
    from tui_gateway.server import _persist_model_switch

    _persist_model_switch(
        SimpleNamespace(
            new_model=new_model,
            target_provider=target_provider,
            base_url=base_url,
        )
    )


def _model(home) -> dict:
    cfg = yaml.safe_load((home / "config.yaml").read_text()) or {}
    model = cfg.get("model")
    return model if isinstance(model, dict) else {}


class TestPersistUserDeclaredProvider:
    """A target that declares its own endpoint keeps routing."""

    @pytest.mark.parametrize(
        "target_provider",
        [
            "custom:my-litellm",  # what switch_model() actually emits
            "my-litellm",  # display-name spelling, for callers that pass it
            "custom",  # bare: self-heals to the first valid entry
        ],
    )
    def test_endpoint_is_written_when_result_has_none(self, config_home, target_provider):
        """switch_model resolved no URL — take it from the target's own config."""
        _persist(target_provider)

        model = _model(config_home)
        assert model.get("default") == "model-b"
        assert model.get("provider") == target_provider
        assert model.get("base_url") == LITELLM_URL, (
            f"/model cleared base_url for {target_provider!r}; the resolver then "
            "falls back to OpenRouter and the next turn 401s"
        )

    def test_switch_between_custom_providers_uses_the_new_one(self, config_home):
        _persist("custom:my-vllm")

        assert _model(config_home).get("base_url") == VLLM_URL

    def test_explicit_result_base_url_always_wins(self, config_home):
        _persist("custom:my-litellm", base_url="http://192.168.1.10:4001")

        assert _model(config_home).get("base_url") == "http://192.168.1.10:4001"


class TestPersistBuiltInProvider:
    """Built-ins keep the historical clear — the registry supplies the URL."""

    def test_switch_to_built_in_clears_stale_base_url(self, config_home):
        _persist("anthropic", new_model="claude-haiku-4-5")

        model = _model(config_home)
        assert model.get("provider") == "anthropic"
        assert model.get("base_url") in ("", None), (
            f"stale custom URL survived a built-in switch: {model.get('base_url')!r}"
        )

    def test_unknown_provider_clears_rather_than_guessing(self, config_home):
        """An unrecognised name declares no endpoint, so don't keep the old one."""
        _persist("not-a-configured-provider")

        assert _model(config_home).get("base_url") in ("", None)
