"""Tool-search stemming loads only snowballstemmer's English stemmer (not all ~30 languages)."""

import json
import subprocess
import sys
from pathlib import Path

import snowballstemmer

from tools import tool_search_catalog as catalog

REPO = Path(__file__).resolve().parents[2]
WORDS = ("issues running created generously connection databases organization flies studies agreed "
         "searching repositories scheduling deployments").split()


def test_stems_match_the_public_english_stemmer():
    reference = snowballstemmer.stemmer("english")
    assert [catalog._stem(w) for w in WORDS] == [reference.stemWord(w) for w in WORDS]


def test_stemming_imports_no_other_language():
    code = (
        "import json, sys\n"
        "from tools.tool_search_catalog import _tokenize\n"
        "_tokenize('create_issue for the repositories')\n"
        "print(json.dumps(sorted(m for m in sys.modules if 'snowball' in m)))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True,
                          timeout=120, check=True)
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "snowballstemmer" not in loaded
    assert not [m for m in loaded if m.endswith("_stemmer") and not m.endswith(".english_stemmer")], loaded


def test_falls_back_to_the_public_api_when_the_layout_differs(monkeypatch):
    real_find_spec = catalog.importlib.util.find_spec
    monkeypatch.setattr(catalog.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "snowballstemmer" else real_find_spec(name, *a, **k))
    catalog._english_stemmer_factory.cache_clear()
    try:
        assert catalog._english_stemmer_factory()().stemWord("issues") == "issu"
    finally:
        catalog._english_stemmer_factory.cache_clear()


def test_falls_back_when_the_module_lacks_the_class(monkeypatch):
    import types
    monkeypatch.setattr(catalog.importlib, "import_module", lambda name, *a, **k: types.ModuleType(name))
    catalog._english_stemmer_factory.cache_clear()
    try:
        assert catalog._english_stemmer_factory()().stemWord("issues") == "issu"
    finally:
        catalog._english_stemmer_factory.cache_clear()


def test_errors_inside_the_stemmer_module_are_not_swallowed(monkeypatch):
    import pytest

    def broken(name, *a, **k):
        raise AttributeError("bug inside english_stemmer.py")

    monkeypatch.setattr(catalog.importlib, "import_module", broken)
    catalog._english_stemmer_factory.cache_clear()
    try:
        with pytest.raises(AttributeError, match="bug inside"):
            catalog._english_stemmer_factory()
    finally:
        catalog._english_stemmer_factory.cache_clear()
