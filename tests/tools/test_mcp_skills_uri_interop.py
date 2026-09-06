"""SEP skill URI prefixes are optional; the authority is not a network host."""
from __future__ import annotations

import hashlib

import pytest

from tools.mcp_skills_protocol import validate_skill_entry
from tools.mcp_skills_registry import relative_resource_path


def entry(root):
    name = 'portable-demo'
    body = f'---\nname: {name}\ndescription: Portable example\n---\nRead the reference.\n'.encode()
    files = [('SKILL.md', body), ('references/info.md', b'Portable reference')]
    return {
        'uri': root + '/SKILL.md',
        'frontmatter': {'name': name, 'description': 'Portable example'},
        'resources': [
            {'uri': root + '/' + path, 'digest': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
            for path, raw in files
        ],
    }


@pytest.mark.parametrize('root', [
    'skill://portable-demo',
    'skill://examples/portable-demo',
    'skill://examples/team/portable-demo',
    'github://owner/repo/skills/portable-demo',
])
def test_optional_prefix_preserves_exact_identity_and_relative_support_path(root):
    data = entry(root)
    validated = validate_skill_entry(data)
    assert validated.uri == data['uri']
    assert relative_resource_path(data, data['resources'][1]['uri']) == 'references/info.md'


@pytest.mark.parametrize('root', ['skill://portable-demo', 'skill://examples/portable-demo'])
def test_optional_prefix_does_not_allow_different_authority_or_name(root):
    data = entry(root)
    data['resources'][1]['uri'] = 'skill://other/references/info.md'
    with pytest.raises(ValueError, match='escapes'):
        validate_skill_entry(data)
    data = entry(root)
    data['frontmatter']['name'] = 'different-name'
    with pytest.raises(ValueError, match='name'):
        validate_skill_entry(data)


@pytest.mark.parametrize("resource_uri", [
    "skill://user@portable-demo/references/note.md",
    "skill://portable-demo/references/note.md#fragment",
    "skill://portable-demo/references/note.md?changed=query",
    "skill://portable-demo-sibling/references/note.md",
    "skill://portable-demo/references/%252e%252e/note.md",
])
def test_authority_only_skill_preserves_uri_rejections(resource_uri):
    data = entry("skill://portable-demo")
    data["resources"][1]["uri"] = resource_uri
    with pytest.raises(ValueError):
        validate_skill_entry(data)
