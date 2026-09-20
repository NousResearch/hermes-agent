"""Model visibility has to survive the round-trip through config, for every surface.

The picker roster used to live only in the Desktop renderer's localStorage, so a
second client — a phone, a TUI, another window on the same backend — could not read
it and showed a list the operator never chose. Moving it into `display.visible_models`
only helps if `config.set` writes exactly what `config.get` reads back, including the
two states that are easy to collapse into each other:

- absent  -> "never customised", the client applies its own curated default
- `[]`    -> "the operator hid everything", which must NOT be re-seeded
"""

import pytest
import yaml

from tui_gateway import server


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Point the server's config read/write at a temp file."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield tmp_path / "config.yaml"
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _set(value):
    return server._methods["config.set"](1, {"key": "visible_models", "value": value})


def _get():
    return server._methods["config.get"](1, {"key": "visible_models"})


def test_a_roster_survives_the_round_trip(config_home):
    keys = ["anthropic::claude-opus-5", "or::auto/best-coding"]

    assert _set(keys)["result"] == {"key": "visible_models", "value": keys}
    assert yaml.safe_load(config_home.read_text())["display"]["visible_models"] == keys
    assert _get()["result"] == {"value": keys}


def test_an_uncustomised_backend_answers_null_not_empty(config_home):
    """`null` and `[]` are different answers; only `null` may be re-seeded."""
    assert _get()["result"] == {"value": None}


def test_hiding_everything_is_preserved_as_empty(config_home):
    """An empty roster is a decision. Reading it back as `null` would make the
    client helpfully restore the models the operator just switched off."""
    assert _set([])["result"] == {"key": "visible_models", "value": []}
    assert _get()["result"] == {"value": []}


def test_clearing_with_null_returns_to_uncustomised(config_home):
    _set(["anthropic::claude-opus-5"])

    assert _set(None)["result"] == {"key": "visible_models", "value": None}
    assert _get()["result"] == {"value": None}


def test_a_provider_sentinel_is_a_valid_key(config_home):
    """`provider::` means "this provider is fully hidden" — rejecting it would make
    the picker re-add that provider's defaults on the next open."""
    assert _set(["copilot::"])["result"] == {"key": "visible_models", "value": ["copilot::"]}
    assert _get()["result"] == {"value": ["copilot::"]}


def test_duplicates_collapse_so_the_roster_cannot_grow_unbounded(config_home):
    answer = _set(["anthropic::claude-opus-5", "anthropic::claude-opus-5"])

    assert answer["result"]["value"] == ["anthropic::claude-opus-5"]


def test_a_cli_string_is_accepted_as_json_or_comma_separated(config_home):
    """`hermes config set` hands the value over as text, not as a list."""
    assert _set('["anthropic::claude-opus-5"]')["result"]["value"] == ["anthropic::claude-opus-5"]
    assert _set("a::b, c::d")["result"]["value"] == ["a::b", "c::d"]


def test_a_malformed_key_is_refused_rather_than_written(config_home):
    answer = _set(["not-a-model-key"])

    assert answer["error"]["code"] == 4002
    assert not config_home.exists()


def test_a_scalar_on_disk_reads_back_as_uncustomised(config_home):
    """A hand-edited string must not read back as a roster of single characters."""
    config_home.write_text(yaml.safe_dump({"display": {"visible_models": "claude"}}))
    server._cfg_cache = server._cfg_sig = server._cfg_path = None

    assert _get()["result"] == {"value": None}


def test_non_string_entries_are_refused(config_home):
    assert _set(["anthropic::claude-opus-5", 7])["error"]["code"] == 4002
