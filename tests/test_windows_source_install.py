"""Source archive constraints for Windows pip installations."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_python_source_archive_excludes_the_deep_website_tree():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "website/ export-ignore" in attributes
