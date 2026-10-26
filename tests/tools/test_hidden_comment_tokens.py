"""Hidden-comment token regressions for external context files (#48809)."""

import pytest

from agent.prompt_builder import _scan_context_content
from tools.threat_patterns import scan_for_threats


@pytest.mark.parametrize('token', ['system_prompt', 'ignore_rules', 'override_policy', 'secret_value', 'hidden_note'])
def test_underscore_tokens_are_blocked(token):
    content = f'<!-- {token} -->'
    assert 'html_comment_injection' in scan_for_threats(content)
    assert 'BLOCKED' in _scan_context_content(content, 'external.md')


def test_benign_map_comment_is_preserved():
    content = '<!-- ECOSYSTEM-MAP: workspace navigation -->'
    assert 'html_comment_injection' not in scan_for_threats(content)
    assert _scan_context_content(content, 'external.md') == content
