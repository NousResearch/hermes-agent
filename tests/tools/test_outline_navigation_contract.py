"""Navigation contract: discover an unknown section, then read its actual body."""

import json
import base64

import pytest

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


@pytest.mark.parametrize("line_count,target,body", [
    (8000, 619, "body"),
    (12000, 120, "detail " * 12),
    (12000, 10000, "detail " * 12),
])
def test_navigation_tradeoffs(tmp_path, monkeypatch, record_property, line_count, target, body):
    """Counts tool turns/content, not LLM tokens, latency, or physical disk I/O."""
    path = tmp_path / "spec.md"
    lines = [body] * line_count
    lines[0] = "# Overview"
    lines[target - 1:target + 1] = ["## Acceptance", "Ready"]
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    results = {}
    for strategy in ("pagination", "outline", "known_keyword", "complete_outline"):
        ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda _: ops)
        task = f"comparison-{strategy}"
        file_tools.clear_file_ops_cache(task)
        counts = {"calls": 0, "result_chars": 0, "outline_payload_bytes": 0}
        reader = ops.read_outline_window

        def counted_window(*args):
            window = reader(*args)
            counts["outline_payload_bytes"] += len(base64.b64decode(window["data"]))
            return window

        monkeypatch.setattr(ops, "read_outline_window", counted_window)

        def invoke(tool="read_file", **arguments):
            raw = registry.dispatch(tool, {"path": str(path), **arguments}, task_id=task)
            counts["calls"] += 1
            counts["result_chars"] += len(raw.replace(str(path), "spec.md"))
            result = json.loads(raw)
            assert "error" not in result, result
            return result

        if strategy == "pagination":
            for offset in range(1, line_count + 1, 500):
                page = invoke(offset=offset, limit=500)
                if "Ready" in page["content"]:
                    break
            else:
                pytest.fail("pagination did not find the section")
        elif strategy == "known_keyword":
            found = invoke("search_files", pattern="^## Acceptance$", limit=1)
            section = invoke(offset=found["matches"][0]["line"], limit=2)
            assert "Ready" in section["content"]
        else:
            cursor = None
            for _ in range(100):
                page = invoke(mode="outline", cursor=cursor)
                selected = next((e for e in page["outline"] if e["heading"] == "Acceptance"), None)
                if selected and strategy == "outline":
                    assert "Ready" in invoke(offset=selected["line"], limit=2)["content"]
                    break
                if page["scan_complete"]:
                    assert strategy == "complete_outline"
                    break
                cursor = page["next_cursor"]
            else:
                pytest.fail("outline did not terminate")
        results[strategy] = counts
    # Known-keyword search is intentionally a strong baseline, not a straw man.
    assert results["known_keyword"]["calls"] == 2
    assert results["outline"]["result_chars"] < results["pagination"]["result_chars"]
    assert results["complete_outline"]["outline_payload_bytes"] == len(data)
    record_property("navigation_metrics", json.dumps({
        "file_bytes": len(data), "target_line": target, "strategies": results}))
