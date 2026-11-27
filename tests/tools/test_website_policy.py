"""A blocklist rule written as a URL or ``host:port`` / ``user@host`` must block that host.

Requests are matched by bare hostname, so a rule that keeps its port or userinfo never matched
and the policy silently failed open.
"""

import pytest

from tools.website_policy import check_website_access


def _policy(tmp_path, rule):
    path = tmp_path / "config.yaml"
    path.write_text(
        "security:\n  website_blocklist:\n    enabled: true\n    domains:\n"
        f"      - '{rule}'\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "rule, blocked_url",
    [
        ("evil.com:8443", "https://evil.com:8443/a"),
        ("https://evil.com:8443/x", "https://sub.evil.com/"),
        ("user@evil.com", "https://evil.com/"),
        ("https://user:pw@evil.com:8443/", "https://evil.com/"),
        ("*.evil.com", "https://sub.evil.com/"),
        ("http://[::1]:8080/", "http://[::1]:9000/"),
    ],
)
def test_rule_with_port_or_userinfo_blocks_its_host(tmp_path, rule, blocked_url):
    config = _policy(tmp_path, rule)

    block = check_website_access(blocked_url, config_path=config)

    assert block is not None
    assert check_website_access("https://notevil.com/", config_path=config) is None
