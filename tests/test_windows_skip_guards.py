"""Regression: tests importing UNIX-only stdlib modules must use
``pytest.importorskip`` so collection on Windows doesn't hard-fail
before any test runs (#22420).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_imports(body: str) -> list[str]:
    """Return only the unindented import lines (module-level)."""
    return [
        line for line in body.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]


def test_gateway_service_uses_importorskip_for_pwd():
    body = _read(REPO_ROOT / "tests" / "hermes_cli" / "test_gateway_service.py")
    top = _top_level_imports(body)
    assert "import pwd" not in top, (
        "test_gateway_service.py must not `import pwd` at module top — "
        "use `pwd = pytest.importorskip('pwd')` so Windows can skip the file."
    )
    assert 'pytest.importorskip("pwd")' in body


def test_file_sync_back_uses_importorskip_for_fcntl():
    body = _read(REPO_ROOT / "tests" / "tools" / "test_file_sync_back.py")
    top = _top_level_imports(body)
    assert "import fcntl" not in top, (
        "test_file_sync_back.py must not `import fcntl` at module top — "
        "use `fcntl = pytest.importorskip('fcntl')` so Windows can skip the file."
    )
    assert 'pytest.importorskip("fcntl")' in body
