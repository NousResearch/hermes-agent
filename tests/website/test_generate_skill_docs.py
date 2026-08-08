"""Tests for website/scripts/generate-skill-docs.py.

The generator turns every `skills/**/SKILL.md` into a Docusaurus page before
the `docs-site-checks` CI workflow runs `ascii-guard lint` on the result. If
a SKILL.md contains ASCII diagrams (box-drawing chars in a fenced code block)
without its own `<!-- ascii-guard-ignore -->` markers, the generator must
add them defensively — otherwise every PR touching `website/**` fails lint
on unrelated skill content.

Regression for issue #15305.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "website" / "scripts" / "generate-skill-docs.py"


@pytest.fixture(scope="module")
def gen_module():
    """Load generate-skill-docs.py as a module (hyphenated filename, not importable via normal import)."""
    spec = importlib.util.spec_from_file_location("generate_skill_docs", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_code_block_without_box_chars_is_not_wrapped(gen_module):
    """Plain bash/python code blocks should stay uncluttered."""
    body = "Intro.\n\n```bash\npip install foo\nfoo --run\n```\n\nOutro."
    result = gen_module.mdx_escape_body(body)
    assert "ascii-guard-ignore" not in result
    assert "pip install foo" in result


def test_code_block_with_box_chars_gets_wrapped(gen_module):
    """A code fence containing Unicode box-drawing chars must be wrapped in
    ascii-guard-ignore comments so the docs-site-checks lint can't fail on
    a skill's own diagram (issue #15305)."""
    body = (
        "Some text.\n\n"
        "```\n"
        "┌─────────┐\n"
        "│ diagram │\n"
        "└─────────┘\n"
        "```\n\n"
        "More text."
    )
    result = gen_module.mdx_escape_body(body)
    assert "<!-- ascii-guard-ignore -->" in result
    assert "<!-- ascii-guard-ignore-end -->" in result
    # The wrapper must sit OUTSIDE the fence, not inside.
    wrap_open = result.index("<!-- ascii-guard-ignore -->")
    fence_open = result.index("```\n┌")
    assert wrap_open < fence_open


def test_multiple_code_blocks_only_box_ones_wrapped(gen_module):
    """Mixed body: plain code stays plain, box code gets wrapped."""
    body = (
        "```bash\necho hi\n```\n\n"
        "```\n┌──┐\n│  │\n└──┘\n```\n\n"
        "```python\nprint('ok')\n```"
    )
    result = gen_module.mdx_escape_body(body)
    # exactly one wrap pair
    assert result.count("<!-- ascii-guard-ignore -->") == 1
    assert result.count("<!-- ascii-guard-ignore-end -->") == 1
    # plain blocks untouched
    assert "echo hi" in result
    assert "print('ok')" in result


def test_tilde_fenced_box_is_wrapped(gen_module):
    """The generator supports both ``` and ~~~ fences — both must be covered."""
    body = "~~~\n│ box │\n~~~"
    result = gen_module.mdx_escape_body(body)
    assert "<!-- ascii-guard-ignore -->" in result


def test_already_wrapped_source_double_wraps_harmlessly(gen_module):
    """If the SKILL.md already has ascii-guard-ignore markers, the generator's
    extra wrap is harmless (ascii-guard tolerates adjacent duplicate markers).
    The test just verifies we don't crash and the content survives."""
    body = (
        "<!-- ascii-guard-ignore -->\n"
        "```\n┌─┐\n└─┘\n```\n"
        "<!-- ascii-guard-ignore-end -->"
    )
    result = gen_module.mdx_escape_body(body)
    assert "┌─┐" in result
    # At least one marker pair survives
    assert "<!-- ascii-guard-ignore -->" in result
    assert "<!-- ascii-guard-ignore-end -->" in result


def test_box_drawing_detection_covers_common_chars(gen_module):
    """Smoke-test that the char set covers box-drawing ranges actually used
    in skill diagrams."""
    # Sample from real SKILL.md diagrams (segment-anything, research-paper-writing, etc.)
    for ch in "┌┐└┘─│├┤┬┴┼═║╔╗╚╝╭╮╯╰▶◀▲▼":
        assert ch in gen_module._BOX_DRAWING_CHARS, f"missing: {ch!r}"


def test_bundled_catalog_explains_missing_local_skills(gen_module):
    """The bundled catalog should explain how to restore a listed skill that
    was removed from the local profile's skills tree."""
    result = gen_module.build_catalog_md_bundled([])
    assert "respects local deletions and user edits" in result
    assert "hermes skills reset <name> --restore" in result


def test_mdx_escape_body_escapes_mdx_syntax_without_touching_safe_markup(gen_module):
    body = (
        "Use {value} and } outside code.\n"
        "Inline `{raw}` and `<Unsafe>` are preserved.\n"
        "<!-- keep {comment} and <tag> -->\n"
        '<br><br/><img src="icon.png" alt="icon"><a href="/docs">safe</a>\n'
        '<Widget data={value}>child</Widget>'
    )

    result = gen_module.mdx_escape_body(body)

    assert "Use &#123;value&#125; and &#125; outside code." in result
    assert "`{raw}`" in result
    assert "`<Unsafe>`" in result
    assert "<!-- keep {comment} and <tag> -->" in result
    assert '<br><br/><img src="icon.png" alt="icon"><a href="/docs">safe</a>' in result
    assert "&lt;Widget data=&#123;value&#125;>child&lt;/Widget>" in result


def test_rewrite_relative_links_rewrites_only_repo_relative_targets(gen_module):
    meta = {
        "source_kind": "bundled",
        "rel_path": "productivity/notes",
    }
    body = (
        "[local](references/setup.md)\n"
        "[dot](./templates/example.md)\n"
        "[absolute](https://example.com/x.md)\n"
        "[anchor](#usage)\n"
        "[mailto](mailto:team@example.com)\n"
        "[root](/docs/user-guide)\n"
        "[protocol](//cdn.example.com/file.md)"
    )

    result = gen_module.rewrite_relative_links(body, meta)

    base = (
        "https://github.com/NousResearch/hermes-agent/blob/main/"
        "skills/productivity/notes"
    )
    assert f"[local]({base}/references/setup.md)" in result
    assert f"[dot]({base}/templates/example.md)" in result
    assert "[absolute](https://example.com/x.md)" in result
    assert "[anchor](#usage)" in result
    assert "[mailto](mailto:team@example.com)" in result
    assert "[root](/docs/user-guide)" in result
    assert "[protocol](//cdn.example.com/file.md)" in result


def test_parse_skill_md_reads_frontmatter_and_body(gen_module, tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: test-skill\n"
        "description: Test skill description.\n"
        "---\n"
        "\n"
        "# Test Skill\n"
        "\n"
        "Body text.\n",
        encoding="utf-8",
    )

    parsed = gen_module.parse_skill_md(skill_md)

    assert parsed["frontmatter"] == {
        "name": "test-skill",
        "description": "Test skill description.",
    }
    assert parsed["body"].startswith("# Test Skill")
    assert parsed["body"].endswith("Body text.\n")


def test_parse_skill_md_requires_frontmatter_delimiter(gen_module, tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("name: missing-frontmatter\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no frontmatter"):
        gen_module.parse_skill_md(skill_md)


def test_sanitize_yaml_string_escapes_and_collapses_whitespace(gen_module):
    raw = r' C:\Tools "Name"' + "\n\tmore  text "

    result = gen_module.sanitize_yaml_string(raw)

    assert result == r'C:\\Tools \"Name\" more text'


def test_derive_skill_meta_handles_supported_layouts(gen_module, tmp_path):
    source_dir = tmp_path / "skills"

    cases = [
        (("solo",), {"category": "solo", "sub": None, "slug": "solo"}),
        (("writing", "research"), {"category": "writing", "sub": None, "slug": "research"}),
        (
            ("dev", "languages", "python"),
            {"category": "dev", "sub": "languages", "slug": "python"},
        ),
    ]

    for rel_parts, expected in cases:
        skill_path = source_dir.joinpath(*rel_parts, "SKILL.md")
        meta = gen_module.derive_skill_meta(skill_path, source_dir, "bundled")
        assert meta["source_kind"] == "bundled"
        assert meta["rel_path"] == str(Path(*rel_parts))
        for key, value in expected.items():
            assert meta[key] == value


def test_derive_skill_meta_rejects_deep_layouts(gen_module, tmp_path):
    source_dir = tmp_path / "skills"
    skill_path = source_dir / "a" / "b" / "c" / "d" / "SKILL.md"

    with pytest.raises(ValueError, match="Unexpected skill layout"):
        gen_module.derive_skill_meta(skill_path, source_dir, "bundled")


def test_page_helpers_build_doc_paths_and_ids(gen_module):
    meta = {
        "source_kind": "bundled",
        "category": "dev",
        "sub": None,
        "slug": "python",
        "rel_path": "dev/python",
    }
    sub_meta = {
        "source_kind": "optional",
        "category": "dev",
        "sub": "languages",
        "slug": "python",
        "rel_path": "dev/languages/python",
    }

    assert gen_module.page_id(meta) == "dev-python"
    assert gen_module.page_id(sub_meta) == "dev-languages-python"
    assert (
        gen_module.page_output_path(meta)
        == gen_module.SKILLS_PAGES / "bundled" / "dev" / "dev-python.md"
    )
    assert (
        gen_module.sidebar_doc_id(sub_meta)
        == "user-guide/skills/optional/dev/dev-languages-python"
    )


def test_render_skill_page_minimal_frontmatter(gen_module):
    meta = {
        "source_kind": "bundled",
        "category": "guides",
        "sub": None,
        "slug": "tiny-skill",
        "rel_path": "guides/tiny-skill",
    }
    body = "Use {value} with [setup](references/setup.md)."

    page = gen_module.render_skill_page(meta, {}, body)

    assert page.startswith("---\n")
    assert 'sidebar_label: "Tiny Skill"' in page
    assert 'description: "tiny-skill"' in page
    assert "# Tiny Skill" in page
    assert "| Source | Bundled (installed by default) |" in page
    assert "| Path | `skills/guides/tiny-skill` |" in page
    assert "Use &#123;value&#125; with [setup](" in page
    assert (
        "https://github.com/NousResearch/hermes-agent/blob/main/"
        "skills/guides/tiny-skill/references/setup.md"
    ) in page


def test_render_skill_page_includes_optional_metadata_and_related_links(gen_module):
    meta = {
        "source_kind": "optional",
        "category": "ops",
        "sub": None,
        "slug": "ops-helper",
        "rel_path": "ops/ops-helper",
    }
    related_meta = {
        "source_kind": "bundled",
        "category": "dev",
        "sub": None,
        "slug": "other-skill",
        "rel_path": "dev/other-skill",
    }
    fm = {
        "name": "ops-helper",
        "description": "Does useful work. Extra detail stays out of the summary.",
        "version": "1.2.3",
        "author": "Docs Team",
        "license": "MIT",
        "dependencies": ["python", "git"],
        "platforms": ["linux", "darwin"],
        "metadata": {
            "hermes": {
                "tags": ["ops", "docs"],
                "related_skills": ["other-skill", "missing-skill"],
            }
        },
    }

    page = gen_module.render_skill_page(
        meta,
        fm,
        "<p>Safe body</p>",
        skill_index={"other-skill": related_meta},
    )

    assert 'description: "Does useful work"' in page
    assert "# Ops Helper" in page
    assert "Optional" in page
    assert "hermes skills install official/ops/ops-helper" in page
    assert "| Version | `1.2.3` |" in page
    assert "| Author | Docs Team |" in page
    assert "| License | MIT |" in page
    assert "| Dependencies | `python`, `git` |" in page
    assert "| Platforms | linux, darwin |" in page
    assert "| Tags | `ops`, `docs` |" in page
    assert "[`other-skill`](/docs/user-guide/skills/bundled/dev/dev-other-skill)" in page
    assert "`missing-skill`" in page
    assert "<p>Safe body</p>" in page


def test_build_optional_catalog_empty_has_guidance(gen_module):
    catalog = gen_module.build_catalog_md_optional([])

    assert "# Optional Skills Catalog" in catalog
    assert "hermes skills install official/<category>/<skill>" in catalog
    assert "## Contributing Optional Skills" in catalog
    assert "| Skill | Description |" not in catalog


def test_build_optional_catalog_includes_one_entry(gen_module):
    meta = {
        "source_kind": "optional",
        "category": "mlops",
        "sub": None,
        "slug": "flash-attention",
        "rel_path": "mlops/flash-attention",
    }
    parsed = {
        "frontmatter": {
            "name": "flash-attention",
            "description": "Fast | reliable\nsecond line.",
        }
    }

    catalog = gen_module.build_catalog_md_optional([(meta, parsed)])

    assert "## mlops" in catalog
    assert "| Skill | Description |" in catalog
    assert (
        "| [**flash-attention**](/docs/user-guide/skills/optional/mlops/"
        "mlops-flash-attention) | Fast \\| reliable second line. |"
    ) in catalog


def test_build_sidebar_items_groups_entries_by_source_and_category(gen_module):
    bundled = {
        "source_kind": "bundled",
        "category": "dev",
        "sub": None,
        "slug": "python",
        "rel_path": "dev/python",
    }
    optional = {
        "source_kind": "optional",
        "category": "dev",
        "sub": "languages",
        "slug": "rust",
        "rel_path": "dev/languages/rust",
    }

    tree = gen_module.build_sidebar_items([(bundled, {}), (optional, {})])

    assert tree["bundled_categories"] == [
        {
            "type": "category",
            "label": "dev",
            "key": "skills-bundled-dev",
            "collapsed": True,
            "items": ["user-guide/skills/bundled/dev/dev-python"],
        }
    ]
    assert tree["optional_categories"] == [
        {
            "type": "category",
            "label": "dev",
            "key": "skills-optional-dev",
            "collapsed": True,
            "items": ["user-guide/skills/optional/dev/dev-languages-rust"],
        }
    ]


def test_render_sidebar_item_outputs_nested_typescript(gen_module):
    item = {
        "label": "dev",
        "key": "skills-bundled-dev",
        "collapsed": True,
        "items": ["user-guide/skills/bundled/dev/dev-python"],
    }

    lines = gen_module._render_sidebar_item(item, 2)

    assert lines == [
        "  {",
        "    type: 'category',",
        "    label: 'dev',",
        "    key: 'skills-bundled-dev',",
        "    collapsed: true,",
        "    items: [",
        "      'user-guide/skills/bundled/dev/dev-python',",
        "    ],",
        "  },",
    ]
