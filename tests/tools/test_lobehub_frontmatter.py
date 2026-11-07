"""Catalog text remains string data when a LobeHub agent becomes a skill."""

import httpx
import pytest

from tools import skills_hub
from tools.skills_hub_models import _parse_frontmatter
from tools.skills_hub_sources import LobeHubSource


@pytest.mark.parametrize("identifier,description", [
    ("helper", "Help with code: Python"), ("helper", "Keep # literal text"),
    ("helper", "First line\n---\nSecond line"), ("yes", "Normal description"),
    ("helper", "介绍：" + "x" * 600),
])
def test_fetched_lobehub_skill_preserves_catalog_metadata(monkeypatch, identifier, description):
    tags = ["yes", "null", "2026-01-01", "a,b", "[brackets]", "中文"]
    agent = {"identifier": identifier, "meta": {"description": description, "tags": tags},
             "config": {"systemRole": "Answer carefully."}}
    monkeypatch.setattr(skills_hub, "_skills_hub_http_get",
        lambda url, **kwargs: httpx.Response(200, json=agent, request=httpx.Request("GET", url)))
    bundle = LobeHubSource().fetch(f"lobehub/{identifier}")
    assert bundle is not None
    metadata = _parse_frontmatter(bundle.files["SKILL.md"])
    assert metadata["name"] == bundle.name
    assert metadata["description"] == description[:500]
    assert metadata["metadata"]["hermes"]["tags"] == tags
    assert "## Instructions\n\nAnswer carefully." in bundle.files["SKILL.md"]
    assert description in bundle.files["SKILL.md"]
