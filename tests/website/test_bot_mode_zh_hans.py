"""Regression for issue #94366.

The zh-Hans docs tree had no translation of the Bot Mode user guide, so
zh-Hans readers silently fell back to the English page. These checks are
invariants of the translation, not a mirror of the English page: an
English-only edit (new heading, new code block) must not turn main red.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EN_DOC = REPO_ROOT / "website" / "docs" / "user-guide" / "bot-mode.md"
ZH_DOC = (
    REPO_ROOT
    / "website"
    / "i18n"
    / "zh-Hans"
    / "docusaurus-plugin-content-docs"
    / "current"
    / "user-guide"
    / "bot-mode.md"
)

FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
# Front matter keys Docusaurus uses for routing; they must agree between the
# source and its translation or the zh-Hans page lands on another URL.
ROUTING_KEYS = ("id", "slug")


def _front_matter(text: str) -> dict[str, str]:
    match = FRONT_MATTER_RE.match(text)
    assert match, "page must start with a front matter block"
    pairs = (line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def _machine_readable_lines(block: str) -> tuple[str, ...]:
    # Comments are translated; commands, paths, config keys and payloads are not.
    return tuple(line.split("#", 1)[0].rstrip() for line in block.strip().splitlines())


def test_translation_exists_and_routes_like_the_english_page():
    en_meta = _front_matter(EN_DOC.read_text(encoding="utf-8"))
    zh_meta = _front_matter(ZH_DOC.read_text(encoding="utf-8"))
    assert zh_meta.get("title"), "zh-Hans page needs its own title"
    for key in ROUTING_KEYS:
        assert zh_meta.get(key) == en_meta.get(key), f"front matter '{key}' must match the English page"


def test_code_blocks_are_copied_from_the_english_page():
    en_blocks = {_machine_readable_lines(b) for b in CODE_BLOCK_RE.findall(EN_DOC.read_text(encoding="utf-8"))}
    zh_blocks = [_machine_readable_lines(b) for b in CODE_BLOCK_RE.findall(ZH_DOC.read_text(encoding="utf-8"))]
    assert zh_blocks, "translation lost its code blocks"
    stale = [b for b in zh_blocks if b not in en_blocks]
    assert not stale, (
        "zh-Hans code blocks must be byte-accurate copies of blocks that still exist "
        f"in the English page (translation stale for: {stale})"
    )
