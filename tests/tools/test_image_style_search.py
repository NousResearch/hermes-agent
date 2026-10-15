"""image_style_search: a bounded name search over Krea's public style library, offered only
while the Krea plugin serves image_generate."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools import image_generation_tool as ig


def _catalog(count: int) -> list[dict[str, str]]:
    return [{"id": f"sty_{i}", "name": f"Watercolor {i}"} for i in range(count)]


def test_search_returns_at_most_ten_name_matches_and_the_total():
    library = _catalog(12) + [{"id": "sty_oil", "name": "Oil Paint"}, {"id": "sty_blank", "name": ""}]
    with patch("plugins.image_gen.krea.list_styles", return_value=(library, None)):
        result = json.loads(ig._handle_image_style_search({"query": " WATER "}))

    assert result["total_matches"] == 12
    assert len(result["styles"]) == 10
    assert result["styles"][0] == {"id": "sty_0", "name": "Watercolor 0"}
    assert "note" not in result


def test_search_asks_for_the_managed_model_and_notes_an_empty_library(monkeypatch):
    monkeypatch.setattr(ig, "_managed_model_plugin", lambda: ("krea", "krea-2-large"))
    with patch("plugins.image_gen.krea.list_styles", return_value=([], None)) as list_styles:
        result = json.loads(ig._handle_image_style_search({"query": "water"}))

    list_styles.assert_called_once_with("krea-2-large")
    assert result["styles"] == [] and result["total_matches"] == 0
    assert "no styles for the active Krea 2 model" in result["note"]


def test_search_reports_an_unavailable_library():
    with patch("plugins.image_gen.krea.list_styles", return_value=(None, "no Krea backend configured")):
        result = json.loads(ig._handle_image_style_search({"query": "water"}))

    assert result["error"] == "Krea style library unavailable: no Krea backend configured"


@pytest.mark.parametrize("provider, model, gateway, direct_available, expected", [
    ("nous", "krea-2-medium", True, False, True),
    ("nous", "krea-2-medium", False, True, False),
    ("nous", "fal-ai/flux-2/klein/9b", True, True, False),
    ("krea", None, False, True, True),
    ("krea", None, False, False, False),
    (None, None, True, True, False),
])
def test_search_is_offered_only_while_krea_serves_image_generate(
        monkeypatch, provider, model, gateway, direct_available, expected):
    import plugins.image_gen.krea as krea_mod

    class _Krea:
        def is_available(self):
            return direct_available

    monkeypatch.setattr(ig, "_read_configured_image_provider", lambda: provider)
    monkeypatch.setattr(ig, "_read_configured_image_model", lambda: model)
    monkeypatch.setattr(ig, "_get_plugin_provider", lambda name, force=False: _Krea() if name == "krea" else None)
    monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", lambda: object() if gateway else None)

    assert ig.check_krea_style_search_requirements() is expected
