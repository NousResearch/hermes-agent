"""Navigation contract: discover an unknown section, then read its actual body."""

import json

from tools import file_tools
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


def test_long_document_structure_to_body(tmp_path, monkeypatch):
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _: ops)
    path = tmp_path / "requirements.md"
    lines = ["# Overview"] + ["context"] * 617 + ["## Acceptance", "Ready"] + ["body"] * 7380
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def read(**args):
        return json.loads(registry.dispatch("read_file", {"path": str(path), **args},
                                           task_id="long-document"))

    outline = read(mode="outline")
    assert "outline" in outline, "read_file must offer structure without returning body"
    target = next(item for item in outline["outline"] if item["heading"] == "Acceptance")
    section = read(offset=target["line"], limit=2)
    prefix = read(offset=1, limit=500)
    assert "Ready" in section["content"]
    assert len(json.dumps(outline)) + len(json.dumps(section)) < len(json.dumps(prefix))
    assert outline["total_lines"] == len(lines)
