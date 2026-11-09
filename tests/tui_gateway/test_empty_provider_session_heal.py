"""#132648: a resumed row with a model but no provider, on the TUI reattach rebuild.

The rebuild healed only bare ``custom``, so an empty provider fell through to the config main slot,
which does not serve the row's model: every turn 404ed. The heal must find the entry that served the
row, leave a row on the main slot's own model alone, and let the entry resolve its current endpoint.
"""

from tests.tui_gateway.test_custom_provider_session_persistence import MIMO_KEY, MIMO_URL, _make_agent_with_override

A, B = "https://a.example/v1", "https://b.example/v1"


def _entry(name, url, key, *models):
    return {"name": name, "base_url": url, "api_key": key, "api_mode": "chat_completions",
            "models": {m: {} for m in models}}


def _config(*entries):
    return {"model": {"default": "model-a", "provider": "custom:entry-a"}, "custom_providers": list(entries)}


class TestEmptyProviderHealsFromModel:
    """A resumed model with an empty provider must recover the owning custom
    entry from the model itself, not fall through to the config main slot."""

    def test_empty_provider_heals_from_model(self, monkeypatch):
        # With no provider AND no base_url the only way back to the entry is the
        # model name, and find_custom_provider_identity_by_model only matches an
        # entry that declares the model (``model``/``default_model``/``models``).
        # LEGACY_LIST_CONFIG's entry declares none — it stands for pre-fix rows
        # where the *base_url* is the recovery key — so this case needs an entry
        # that names the model it serves.
        config = {
            "custom_providers": [
                {
                    "name": "mimo-v2.5-pro",
                    "base_url": MIMO_URL,
                    "api_key": MIMO_KEY,
                    "api_mode": "chat_completions",
                    "models": {"mimo-v2.5-pro": {}},
                }
            ]
        }
        override = {
            "model": "mimo-v2.5-pro",
            "provider": None,
            "base_url": None,
            "api_mode": None,
        }

        kwargs = _make_agent_with_override(override, monkeypatch, config)

        assert kwargs["base_url"] == MIMO_URL
        assert kwargs["api_key"] == MIMO_KEY

    def test_empty_provider_pins_owning_entry_not_main_slot(self, monkeypatch):
        # The #132648 shape: the main slot is entry-a, but the resumed row names
        # entry-b's model with an empty provider. Pre-fix the model fell through
        # to the main slot (entry-a), which does not serve it — every turn 404ed.
        config = {
            "model": {"default": "model-a", "provider": "custom:entry-a"},
            "custom_providers": [
                {
                    "name": "entry-a",
                    "base_url": "https://a.example/v1",
                    "api_key": "sk-entry-a",
                    "api_mode": "chat_completions",
                    "models": {"model-a": {}},
                },
                {
                    "name": "entry-b",
                    "base_url": "https://b.example/v1",
                    "api_key": "sk-entry-b",
                    "api_mode": "chat_completions",
                    "models": {"model-b": {}},
                },
            ],
        }
        override = {
            "model": "model-b",
            "provider": None,
            "base_url": None,
            "api_mode": None,
        }

        kwargs = _make_agent_with_override(override, monkeypatch, config)

        assert kwargs["base_url"] == "https://b.example/v1"
        assert kwargs["api_key"] == "sk-entry-b"


def test_an_empty_provider_row_with_the_main_slots_url_resumes_on_the_entry_serving_its_model(monkeypatch):
    config = _config(_entry("entry-a", A, "sk-a", "model-a"), _entry("entry-b", B, "sk-b", "model-b"))
    # The row carries the main slot's URL, stamped by the turns that already went there.
    override = {"model": "model-b", "provider": None, "base_url": A, "api_mode": None}

    kwargs = _make_agent_with_override(override, monkeypatch, config, model_cfg=config["model"])

    assert (kwargs["base_url"], kwargs["api_key"]) == (B, "sk-b")


def test_an_empty_provider_row_reaches_its_entrys_current_endpoint(monkeypatch):
    config = _config(_entry("glm", "https://new.example/v1", "sk-glm", "GLM-5.3"))
    override = {"model": "GLM-5.3", "provider": None, "base_url": "https://old.example/v1", "api_mode": None}

    kwargs = _make_agent_with_override(override, monkeypatch, config, model_cfg=config["model"])

    assert (kwargs["base_url"], kwargs["api_key"]) == ("https://new.example/v1", "sk-glm")
