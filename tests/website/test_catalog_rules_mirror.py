"""The catalog submission guide mirrors the canonical admission rules word for word.

``plugin-catalog/README.md`` is where reviewers and the CI gates point; the docs page is where
submitters read. A rule edited in one and not the other tells authors a different policy than
the one their PR is reviewed against.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_BLOCK = re.compile(r"<!-- admission-rules:start[^>]*-->(.*?)<!-- admission-rules:end -->", re.DOTALL)


def _rules(path: Path) -> str:
    match = _BLOCK.search(path.read_text(encoding="utf-8"))
    assert match, f"{path.relative_to(ROOT)} lost its admission-rules markers"
    return " ".join(match.group(1).split())


def test_docs_page_mirrors_readme_admission_rules():
    readme = _rules(ROOT / "plugin-catalog" / "README.md")
    docs = _rules(ROOT / "website" / "docs" / "developer-guide" / "plugins" / "catalog-submission.md")
    assert re.search(r"\b1\. \*\*", readme)
    assert docs == readme, (
        "website/docs/developer-guide/plugins/catalog-submission.md must carry the exact rule block "
        "from plugin-catalog/README.md (copy everything between the admission-rules markers)"
    )
