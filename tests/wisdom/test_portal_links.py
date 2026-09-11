from urllib.parse import parse_qs, unquote, urlparse

import pytest

from hermes_wisdom.portal_links import portal_item_url
from hermes_wisdom.service import WisdomService


@pytest.mark.parametrize("kind", ["review", "skills"])
def test_links_preserve_opaque_team_and_item_without_guessing_slug(kind):
    org_id = "nas_organisation:opaque/team?&id=other"
    item_id = "item/with ?&#+characters"
    url = portal_item_url("http://localhost:3111/", org_id, kind, item_id, version=7)
    parsed = urlparse(url)
    assert parsed.netloc == "localhost:3111"
    assert parsed.path.startswith(f"/wisdom/{kind}/")
    assert unquote(parsed.path.split("/")[-1]) == item_id
    assert parse_qs(parsed.query) == {"org_id": [org_id], "version": ["7"]}
    assert not parsed.fragment


def test_stable_skill_link_remains_an_exact_version_install_reference():
    url = portal_item_url("https://portal.test", "nas_organisation:opaque", "skills", "skill-1", version=7)
    # Parsing does not require a configured profile or network access.
    service = object.__new__(WisdomService)
    assert service._resolve_install_ref(url) == ("skill-1", 7)
