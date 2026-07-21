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


def _generated_page(gen_module, body: str) -> str:
    return f"---\ntitle: Test\n---\n\n{gen_module.GENERATED_PAGE_MARKER}\n\n{body}\n"


def test_check_reports_drift_without_writing(
    gen_module, tmp_path, monkeypatch, capsys
):
    """Check mode reports missing/outdated/stale outputs and changes nothing."""
    skills_pages = tmp_path / "docs" / "user-guide" / "skills"
    monkeypatch.setattr(gen_module, "SKILLS_PAGES", skills_pages)
    monkeypatch.setattr(gen_module, "REPO", tmp_path)

    outdated = skills_pages / "bundled" / "test" / "outdated.md"
    missing = skills_pages / "bundled" / "test" / "missing.md"
    stale = skills_pages / "optional" / "test" / "stale.md"
    handwritten = skills_pages / "bundled" / "test" / "handwritten.md"
    for path in (outdated, stale, handwritten):
        path.parent.mkdir(parents=True, exist_ok=True)
    outdated.write_text(_generated_page(gen_module, "old"), encoding="utf-8")
    stale.write_text(_generated_page(gen_module, "stale"), encoding="utf-8")
    handwritten.write_text("# Hand-written page\n", encoding="utf-8")

    before = {
        path: path.read_bytes()
        for path in (outdated, stale, handwritten)
    }
    expected_outputs = {
        outdated: _generated_page(gen_module, "new"),
        missing: _generated_page(gen_module, "missing"),
    }

    assert not gen_module.check_generated_outputs(
        expected_outputs, set(expected_outputs)
    )

    output = capsys.readouterr().out
    assert "OUTDATED: docs/user-guide/skills/bundled/test/outdated.md" in output
    assert "MISSING: docs/user-guide/skills/bundled/test/missing.md" in output
    assert "STALE: docs/user-guide/skills/optional/test/stale.md" in output
    assert "handwritten.md" not in output
    assert not missing.exists()
    for path, content in before.items():
        assert path.read_bytes() == content


def test_check_bounds_large_diff_output(
    gen_module, tmp_path, monkeypatch, capsys
):
    """A heavily changed generated page must not flood CI logs."""
    monkeypatch.setattr(gen_module, "SKILLS_PAGES", tmp_path / "skills")
    monkeypatch.setattr(gen_module, "REPO", tmp_path)
    page = tmp_path / "large.md"
    page.write_text("\n".join(f"old-{i}" for i in range(200)), encoding="utf-8")
    expected = "\n".join(f"new-{i}" for i in range(200))

    assert not gen_module.check_generated_outputs({page: expected}, set())

    output = capsys.readouterr().out
    assert "additional diff lines omitted" in output
    assert len(output.splitlines()) < gen_module.MAX_REPORTED_DIFF_LINES + 10


def test_generation_removes_only_owned_stale_pages(
    gen_module, tmp_path, monkeypatch
):
    """Cleanup is marker-gated and preserves hand-written Markdown."""
    skills_pages = tmp_path / "docs" / "user-guide" / "skills"
    monkeypatch.setattr(gen_module, "SKILLS_PAGES", skills_pages)
    monkeypatch.setattr(gen_module, "REPO", tmp_path)

    current = skills_pages / "bundled" / "test" / "current.md"
    stale = skills_pages / "bundled" / "old" / "stale.md"
    handwritten = skills_pages / "bundled" / "old" / "notes.md"
    for path in (current, stale, handwritten):
        path.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(_generated_page(gen_module, "current"), encoding="utf-8")
    stale.write_text(_generated_page(gen_module, "stale"), encoding="utf-8")
    handwritten.write_text("# Keep me\n", encoding="utf-8")

    removed = gen_module._remove_stale_generated_pages({current})

    assert removed == [stale.resolve()]
    assert current.exists()
    assert not stale.exists()
    assert handwritten.read_text(encoding="utf-8") == "# Keep me\n"


def test_generation_refuses_to_overwrite_unowned_expected_page(
    gen_module, tmp_path, monkeypatch, capsys
):
    """An expected output collision with hand-written content fails closed."""
    page = tmp_path / "docs" / "user-guide" / "skills" / "bundled" / "x.md"
    page.parent.mkdir(parents=True)
    page.write_text("# Hand-written page\n", encoding="utf-8")

    monkeypatch.setattr(gen_module, "discover_skills", lambda: [])
    monkeypatch.setattr(
        gen_module,
        "build_expected_outputs",
        lambda entries: ({page: _generated_page(gen_module, "replacement")}, {page}),
    )

    assert gen_module.generate_skill_docs() == 1
    assert page.read_text(encoding="utf-8") == "# Hand-written page\n"
    assert "Refusing to overwrite non-generated page" in capsys.readouterr().out


def test_main_check_flag_is_read_only_mode(gen_module, monkeypatch):
    calls = []
    monkeypatch.setattr(
        gen_module,
        "generate_skill_docs",
        lambda *, check=False: calls.append(check) or 0,
    )

    assert gen_module.main(["--check"]) == 0
    assert calls == [True]
