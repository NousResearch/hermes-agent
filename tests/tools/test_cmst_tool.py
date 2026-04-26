import hashlib
import json

import pytest


def test_cmst_toolset_registers_route_and_load(tmp_path, monkeypatch):
    tree_root = tmp_path / "cmst-skill-tree"
    module_dir = tree_root / "modules" / "debug-helper"
    module_dir.mkdir(parents=True)
    content = "---\nname: debug-helper\ndescription: Use when debugging tests.\n---\n\n# Debug Helper\n"
    skill_file = module_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    (tree_root / "manifest.json").write_text(
        json.dumps(
            {
                "tree": "hermes",
                "modules": {
                    "debug-helper": {
                        "status": "enabled",
                        "path": "modules/debug-helper/SKILL.md",
                        "sha256": sha,
                        "keywords": ["debug", "failing", "test"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"skills": {"cmst_tree_root": str(tree_root)}})

    from model_tools import get_tool_definitions, handle_function_call

    tools = get_tool_definitions(enabled_toolsets=["cmst"], quiet_mode=True)
    assert [tool["function"]["name"] for tool in tools] == ["cmst_load", "cmst_route"]
    route_schema = tools[1]["function"]["parameters"]["properties"]
    assert set(route_schema) == {"task", "limit"}

    route_result = json.loads(handle_function_call("cmst_route", {"task": "debug a failing test"}))
    assert route_result["mode"] == "index"
    assert route_result["candidates"][0]["number"] == 1
    assert set(route_result["candidates"][0]) == {"number", "description", "ref"}
    assert route_result["candidates"][0]["description"] == "Use when debugging tests."
    assert "debug-helper" in route_result["candidates"][0]["ref"]

    load_result = json.loads(handle_function_call("cmst_load", {"ref": route_result["candidates"][0]["ref"]}))
    assert load_result["id"] == "debug-helper"
    assert load_result["content"] == content

    auto_result = json.loads(handle_function_call("cmst_route", {"task": "debug a failing test", "mode": "auto"}))
    assert auto_result["mode"] == "runtime"
    assert auto_result["module"]["id"] == "debug-helper"


@pytest.mark.parametrize(
    "task",
    [
        "fix pytest failure after refactor",
        "fix a bug in session handling",
        "investigate unexpected behavior in CLI",
        "debug a test failure after refactor",
    ],
)
def test_cmst_route_indexes_debug_tasks_with_systematic_debugging_first(tmp_path, monkeypatch, task):
    tree_root = tmp_path / "cmst-skill-tree"
    (tree_root / "modules" / "nano-pdf").mkdir(parents=True)
    (tree_root / "modules" / "systematic-debugging").mkdir(parents=True)
    (tree_root / "manifest.json").write_text(
        json.dumps(
            {
                "tree": "hermes",
                "modules": {
                    "nano-pdf": {
                        "status": "enabled",
                        "path": "modules/nano-pdf/SKILL.md",
                        "sha256": "0" * 64,
                        "keywords": ["nano-pdf", "fix", "pdfs"],
                    },
                    "systematic-debugging": {
                        "status": "enabled",
                        "path": "modules/systematic-debugging/SKILL.md",
                        "sha256": "1" * 64,
                        "keywords": ["systematic-debugging", "bug", "test", "failure", "unexpected", "behavior"],
                        "summary": "Use for bugs, test failures, unexpected behavior, and build failures.",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"skills": {"cmst_tree_root": str(tree_root)}})

    from tools.cmst_tool import cmst_route

    route_result = json.loads(cmst_route(task))
    assert route_result["mode"] == "index"
    assert route_result["candidates"][0]["number"] == 1
    assert set(route_result["candidates"][0]) == {"number", "description", "ref"}
    assert route_result["candidates"][0]["description"] == "Use for bugs, test failures, unexpected behavior, and build failures."
    assert "systematic-debugging" in route_result["candidates"][0]["ref"]
    assert "nano-pdf" not in route_result["candidates"][0]["ref"]

    auto_result = json.loads(cmst_route(task, mode="auto"))
    assert auto_result["module"]["id"] == "systematic-debugging"


def test_cmst_route_pdf_context_ranks_pdf_skill_first(tmp_path, monkeypatch):
    tree_root = tmp_path / "cmst-skill-tree"
    (tree_root / "modules" / "nano-pdf").mkdir(parents=True)
    (tree_root / "modules" / "systematic-debugging").mkdir(parents=True)
    (tree_root / "manifest.json").write_text(
        json.dumps(
            {
                "tree": "hermes",
                "modules": {
                    "nano-pdf": {
                        "status": "enabled",
                        "path": "modules/nano-pdf/SKILL.md",
                        "sha256": "0" * 64,
                        "keywords": ["nano-pdf", "edit", "pdfs", "fix", "typos"],
                        "summary": "Use when editing existing PDF files.",
                    },
                    "systematic-debugging": {
                        "status": "enabled",
                        "path": "modules/systematic-debugging/SKILL.md",
                        "sha256": "1" * 64,
                        "keywords": ["systematic-debugging", "bug", "test", "failure", "unexpected", "behavior"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"skills": {"cmst_tree_root": str(tree_root)}})

    from tools.cmst_tool import cmst_route

    route_result = json.loads(cmst_route("fix a typo in a PDF"))
    assert "nano-pdf" in route_result["candidates"][0]["ref"]
    assert route_result["candidates"][0]["description"] == "Use when editing existing PDF files."


def test_cmst_route_can_return_multiple_relevant_skill_cards(tmp_path, monkeypatch):
    tree_root = tmp_path / "cmst-skill-tree"
    (tree_root / "modules" / "systematic-debugging").mkdir(parents=True)
    (tree_root / "modules" / "test-driven-development").mkdir(parents=True)
    (tree_root / "manifest.json").write_text(
        json.dumps(
            {
                "tree": "hermes",
                "modules": {
                    "systematic-debugging": {
                        "status": "enabled",
                        "path": "modules/systematic-debugging/SKILL.md",
                        "sha256": "1" * 64,
                        "keywords": ["systematic-debugging", "bug", "failure", "unexpected", "behavior"],
                    },
                    "test-driven-development": {
                        "status": "enabled",
                        "path": "modules/test-driven-development/SKILL.md",
                        "sha256": "2" * 64,
                        "keywords": ["test-driven-development", "bugfix", "test-first", "implementation"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"skills": {"cmst_tree_root": str(tree_root)}})

    from tools.cmst_tool import cmst_route

    route_result = json.loads(cmst_route("implement a bug fix with tests after diagnosis"))
    candidate_refs = [candidate["ref"] for candidate in route_result["candidates"]]
    assert any("systematic-debugging" in ref for ref in candidate_refs)
    assert any("test-driven-development" in ref for ref in candidate_refs)


def test_cmst_load_rejects_content_hash_mismatch(tmp_path, monkeypatch):
    tree_root = tmp_path / "cmst-skill-tree"
    module_dir = tree_root / "modules" / "debug-helper"
    module_dir.mkdir(parents=True)
    (module_dir / "SKILL.md").write_text("changed content", encoding="utf-8")
    manifest_sha = "0" * 64
    (tree_root / "manifest.json").write_text(
        json.dumps(
            {
                "tree": "hermes",
                "modules": {
                    "debug-helper": {
                        "status": "enabled",
                        "path": "modules/debug-helper/SKILL.md",
                        "sha256": manifest_sha,
                        "keywords": ["debug"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"skills": {"cmst_tree_root": str(tree_root)}})

    from tools.cmst_tool import cmst_load

    result = json.loads(cmst_load(f"cmst://hermes/debug-helper@sha256:{manifest_sha}"))
    assert result == {"error": "module content hash mismatch"}
