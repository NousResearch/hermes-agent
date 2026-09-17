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


def _nested_chain():
    roots = [
        "skill://fixture/parent",
        "skill://fixture/parent/child",
        "skill://fixture/parent/child/grandchild",
    ]
    names = ["parent", "child", "grandchild"]
    payloads = {
        f"{root}/SKILL.md":
            f"---\nname: {name}\ndescription: nested {name}\n---\n".encode()
        for root, name in zip(roots, names)
    }
    entries = []
    for root, name in zip(roots, names):
        resources = [
            {"uri": uri, "digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for uri, raw in payloads.items() if uri.startswith(root + "/")
        ]
        entries.append(validate_skill_entry({
            "uri": f"{root}/SKILL.md",
            "frontmatter": {"name": name, "description": f"nested {name}"},
            "resources": resources,
        }))
    return entries


def test_canonical_nested_chain_is_visible_and_deepest_owner_wins(tmp_path):
    from tools import mcp_skills_registry as registry
    registry.clear_runtime_state()
    parent, child, grandchild = _nested_chain()
    registry.publish_live_catalog(tmp_path, "fixture", "config", [parent, child, grandchild])
    rows = registry.pin_session(tmp_path, "nested")
    assert [row["frontmatter"]["name"] for row in rows] == ["parent", "child", "grandchild"]
    owner = registry.resolve_catalog_resource("fixture", grandchild.uri, tmp_path, "nested")
    assert owner is not None and owner["uri"] == grandchild.uri
    registry.clear_runtime_state()


def test_canonical_branching_nested_tree_is_visible(tmp_path):
    from tools import mcp_skills_registry as registry
    parent, child, _grandchild = _nested_chain()
    sibling = child.model_copy(deep=True)
    sibling.uri = sibling.uri.replace("/child/", "/sibling/")
    sibling.frontmatter["name"] = "sibling"
    for resource in sibling.resources:
        resource.uri = resource.uri.replace("/child/", "/sibling/")
    parent = parent.model_copy(deep=True)
    parent.resources.extend(resource.model_copy(deep=True) for resource in sibling.resources)
    registry.clear_runtime_state()
    registry.publish_live_catalog(tmp_path, "fixture", "config", [parent, child, sibling])
    rows = registry.pin_session(tmp_path, "branching")
    assert {row["frontmatter"]["name"] for row in rows} == {"parent", "child", "sibling"}
    owner = registry.resolve_catalog_resource("fixture", sibling.uri, tmp_path, "branching")
    assert owner is not None and owner["uri"] == sibling.uri
    registry.clear_runtime_state()


@pytest.mark.parametrize("mutation", ["digest", "missing", "omitted", "duplicate"])
def test_invalid_overlap_sets_fail_closed(tmp_path, mutation):
    from tools import mcp_skills_registry as registry
    parent, child, _grandchild = _nested_chain()
    entries = [parent, child]
    if mutation == "digest":
        child = child.model_copy(deep=True)
        child.resources[0].digest = "sha256:" + "0" * 64
        entries = [parent, child]
    elif mutation == "missing":
        parent = parent.model_copy(deep=True)
        parent.resources = parent.resources[:-1]
        entries = [parent, child]
    elif mutation == "omitted":
        parent = parent.model_copy(deep=True)
        parent.resources = [parent.resources[0]]
        entries = [parent, child]
    else:
        entries = [parent, parent]
    registry.clear_runtime_state()
    registry.publish_live_catalog(tmp_path, "fixture", "config", entries)
    assert registry.pin_session(tmp_path, f"bad-{mutation}") == ()
    blocked = registry.resolve_catalog_resource("fixture", child.uri, tmp_path, f"bad-{mutation}")
    if mutation != "duplicate":
        assert blocked is not None and blocked["metadata_allowed"] is False
    registry.clear_runtime_state()


def test_equal_resource_uri_on_different_servers_is_independent(tmp_path):
    from tools import mcp_skills_registry as registry
    parent, _child, _grandchild = _nested_chain()
    registry.clear_runtime_state()
    registry.publish_live_catalog(tmp_path, "one", "a", [parent])
    registry.publish_live_catalog(tmp_path, "two", "b", [parent])
    rows = registry.pin_session(tmp_path, "cross-server")
    assert {row["server"] for row in rows} == {"one", "two"}
    owner = registry.resolve_catalog_resource("one", parent.uri, tmp_path, "cross-server")
    assert owner is not None and owner["server"] == "one"
    registry.clear_runtime_state()
