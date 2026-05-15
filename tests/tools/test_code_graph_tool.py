import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_tools import get_tool_definitions
from tools.code_graph_tool import build_code_graph, code_graph_context, code_graph_impact


@pytest.fixture(autouse=True)
def _mark_tmp_path_as_project(request, tmp_path):
    if "tmp_path" in request.fixturenames:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'codegraph-fixture'\n", encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_code_graph_indexes_python_symbols_imports_and_calls(tmp_path):
    _write(
        tmp_path / "pkg" / "a.py",
        "import os\n\n"
        "def foo(value):\n"
        "    return value + 1\n\n"
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return foo(1)\n",
    )
    _write(
        tmp_path / "pkg" / "b.py",
        "from pkg.a import foo\n\n"
        "def bar():\n"
        "    return foo(41)\n",
    )
    _write(tmp_path / ".env", "PLACEHOLDER_VALUE=should_not_be_indexed\n")
    _write(tmp_path / "venv" / "ignored.py", "def ignored():\n    pass\n")

    graph = build_code_graph(str(tmp_path))

    assert graph["stats"]["files_indexed"] == 2
    assert graph["repo_path"] == "[REDACTED_ABSOLUTE_PATH]"
    assert graph["repo_name"] == tmp_path.name
    assert graph["truncation"]["nodes_truncated"] is False
    assert graph["stats"]["files_skipped"] >= 2
    symbol_ids = {node["id"] for node in graph["nodes"] if node["kind"] in {"function", "class", "method"}}
    assert "pkg/a.py::foo" in symbol_ids
    assert "pkg/a.py::Greeter" in symbol_ids
    assert "pkg/a.py::Greeter.greet" in symbol_ids
    assert "pkg/b.py::bar" in symbol_ids
    assert any(edge["kind"] == "imports" and edge["source"] == "pkg/b.py" for edge in graph["edges"])
    assert any(
        edge["kind"] == "calls"
        and edge["source"] == "pkg/b.py::bar"
        and edge["target"] == "pkg/a.py::foo"
        and edge["confidence"] == "import-resolved"
        for edge in graph["edges"]
    )
    assert "should_not_be_indexed" not in json.dumps(graph)
    assert graph["skipped_reasons"]["excluded_directory"] >= 1
    assert graph["skipped_reasons"]["excluded_file"] >= 1
    assert len(graph["skipped_samples"]) <= 25
    assert all(not sample["path"].startswith("/") for sample in graph["skipped_samples"])


def test_code_graph_reports_bounded_redacted_skip_metadata_without_content(tmp_path):
    _write(tmp_path / "main.py", "def safe():\n    pass\n")
    for index in range(30):
        _write(
            tmp_path / f"token_cache_{index}.py",
            "def leaked_source_content():\n    return 'abc123456789'\n",
        )

    graph = build_code_graph(str(tmp_path))
    graph_json = json.dumps(graph)

    assert graph["skipped_reasons"]["excluded_file"] == 30
    assert len(graph["skipped_samples"]) == 25
    assert all(sample["reason"] == "excluded_file" for sample in graph["skipped_samples"])
    assert all(sample["path"] == "[REDACTED]" for sample in graph["skipped_samples"])
    assert "token_cache_" not in graph_json
    assert "leaked_source_content" not in graph_json
    assert "abc123456789" not in graph_json


def test_code_graph_context_and_impact_are_bounded_and_source_backed(tmp_path):
    _write(
        tmp_path / "pkg" / "a.py",
        "def foo(value):\n"
        "    return value + 1\n",
    )
    _write(
        tmp_path / "pkg" / "b.py",
        "from pkg.a import foo\n\n"
        "def bar():\n"
        "    return foo(41)\n",
    )
    _write(tmp_path / ".env", "SECRET_TOKEN=abc123456789\n")

    context = code_graph_context(str(tmp_path), "foo")
    assert context["matches"][0]["id"] == "pkg/a.py::foo"
    assert context["matches"][0]["line"] == 1
    assert context["matches"][0]["source_excerpt"] == "def foo(value):"
    assert context["skipped_reasons"]["excluded_file"] == 1
    assert context["skipped_samples"][0]["path"] == "[REDACTED]"

    impact = code_graph_impact(str(tmp_path), "foo")
    impacted_ids = {item["id"] for item in impact["impacted"]}
    assert "pkg/b.py::bar" in impacted_ids
    assert impact["query"]["symbol"] == "foo"
    assert impact["scope"] == "inspect-only evidence; graph does not authorize edits"
    assert impact["skipped_reasons"]["excluded_file"] == 1
    assert "abc123456789" not in json.dumps(context)
    assert "abc123456789" not in json.dumps(impact)


def test_code_graph_redacts_standalone_secret_value_shapes_in_source_excerpts(tmp_path):
    raw_values = [
        "sk-" + "A" * 24,
        "ghp_" + "B" * 24,
        "xoxb-" + "C" * 24,
        "AIza" + "D" * 24,
        "TEST_FAKE_VALUE_" + "E" * 12,
    ]
    _write(
        tmp_path / "safe.py",
        "def reveal(openai='{}', github='{}', slack='{}', google='{}', fake='{}'):\n"
        "    return openai\n".format(*raw_values),
    )

    context = code_graph_context(str(tmp_path), "reveal")
    payload = json.dumps(context)

    for raw_value in raw_values:
        assert raw_value not in payload
    assert "[REDACTED]" in payload


def test_code_graph_redacts_secret_shapes_in_paths_symbols_edges_and_queries(tmp_path):
    root = tmp_path / "safe_project"
    _write(root / "pyproject.toml", "[project]\nname = 'safe-project'\n")
    token_path = "github_pat_" + "P" * 24
    token_symbol = "sk_live_" + "S" * 24
    token_call = "AKIA" + "A" * 16
    _write(
        root / f"{token_path}.py",
        f"def {token_symbol}():\n    pass\n\n"
        f"def {token_call}():\n    return {token_symbol}()\n",
    )

    graph = build_code_graph(str(root))
    context = code_graph_context(str(root), token_symbol)
    impact = code_graph_impact(str(root), token_symbol)
    payload = json.dumps({"graph": graph, "context": context, "impact": impact})

    assert token_path not in payload
    assert token_symbol not in payload
    assert token_call not in payload
    assert "github..." in payload
    assert "sk_liv..." in payload
    assert "AKIA" in payload and "..." in payload


def test_code_graph_does_not_emit_false_positive_cross_file_unique_name_calls(tmp_path):
    _write(tmp_path / "a.py", "def save():\n    pass\n")
    _write(tmp_path / "b.py", "def caller():\n    return save()\n")

    graph = build_code_graph(str(tmp_path))

    calls = [edge for edge in graph["edges"] if edge["kind"] == "calls"]
    assert calls == []
    assert graph["stats"]["unresolved_calls"] >= 1

    impact = code_graph_impact(str(tmp_path), "save")
    assert impact["impacted"] == []
    assert impact["impact_edges"] == []


def test_code_graph_does_not_emit_shadowed_bare_call_edges(tmp_path):
    _write(tmp_path / "a.py", "def save():\n    pass\n\ndef caller(save):\n    return save()\n")
    _write(tmp_path / "b.py", "from a import save\n\ndef caller():\n    save = lambda: None\n    return save()\n")

    graph = build_code_graph(str(tmp_path))

    assert not [edge for edge in graph["edges"] if edge["kind"] == "calls" and edge.get("call") == "save"]
    impact = code_graph_impact(str(tmp_path), "save")
    assert impact["impacted"] == []


def test_code_graph_does_not_emit_rebound_import_alias_call_edges(tmp_path):
    _write(tmp_path / "target.py", "def foo():\n    pass\n")
    _write(
        tmp_path / "main.py",
        "from target import foo\n"
        "foo = lambda: None\n\n"
        "def caller():\n"
        "    return foo()\n",
    )

    graph = build_code_graph(str(tmp_path))

    assert not [
        edge for edge in graph["edges"]
        if edge["kind"] == "calls" and edge["source"] == "main.py::caller" and edge.get("call") == "foo"
    ]
    impact = code_graph_impact(str(tmp_path), "foo")
    assert impact["impacted"] == []
    assert impact["impact_edges"] == []


def test_code_graph_does_not_emit_same_file_rebound_or_deleted_call_edges(tmp_path):
    _write(
        tmp_path / "main.py",
        "def removed():\n"
        "    pass\n\n"
        "del removed\n\n"
        "class Rebound:\n"
        "    pass\n\n"
        "Rebound = object\n\n"
        "def calls_removed():\n"
        "    return removed()\n\n"
        "def calls_rebound():\n"
        "    return Rebound()\n",
    )

    graph = build_code_graph(str(tmp_path))

    assert not [
        edge for edge in graph["edges"]
        if edge["kind"] == "calls" and edge.get("call") in {"removed", "Rebound"}
    ]
    assert code_graph_impact(str(tmp_path), "removed")["impacted"] == []
    assert code_graph_impact(str(tmp_path), "Rebound")["impacted"] == []


def test_code_graph_does_not_emit_bare_call_edges_with_module_wildcard_import(tmp_path):
    _write(tmp_path / "target.py", "def foo():\n    pass\n")
    _write(
        tmp_path / "main.py",
        "from target import *\n\n"
        "def foo():\n"
        "    pass\n\n"
        "def caller():\n"
        "    return foo()\n",
    )

    graph = build_code_graph(str(tmp_path))

    assert not [
        edge for edge in graph["edges"]
        if edge["kind"] == "calls" and edge["source"] == "main.py::caller" and edge.get("call") == "foo"
    ]
    assert code_graph_impact(str(tmp_path), "foo")["impacted"] == []


def test_code_graph_does_not_leak_function_local_import_aliases(tmp_path):
    _write(tmp_path / "target.py", "def foo():\n    pass\n")
    _write(
        tmp_path / "main.py",
        "def imports_locally():\n"
        "    from target import foo\n"
        "    return foo()\n\n"
        "def unrelated(foo):\n"
        "    return foo()\n",
    )

    graph = build_code_graph(str(tmp_path))

    assert not [
        edge for edge in graph["edges"]
        if edge["kind"] == "calls" and edge["source"] == "main.py::unrelated" and edge.get("call") == "foo"
    ]


def test_code_graph_does_not_expose_nested_defs_as_file_scope_targets(tmp_path):
    _write(
        tmp_path / "main.py",
        "def outer():\n"
        "    def hidden():\n"
        "        pass\n"
        "    return hidden()\n\n"
        "def caller():\n"
        "    return hidden()\n",
    )

    graph = build_code_graph(str(tmp_path))

    assert not any(node["id"] == "main.py::hidden" for node in graph["nodes"])
    assert not [edge for edge in graph["edges"] if edge["kind"] == "calls" and edge.get("call") == "hidden"]


def test_code_graph_resolves_same_file_bare_calls_only_when_source_backed(tmp_path):
    _write(tmp_path / "a.py", "def save():\n    pass\n\ndef caller():\n    return save()\n")
    _write(tmp_path / "b.py", "def save():\n    pass\n")

    graph = build_code_graph(str(tmp_path))

    assert any(
        edge["kind"] == "calls"
        and edge["source"] == "a.py::caller"
        and edge["target"] == "a.py::save"
        and edge["confidence"] == "same-file"
        for edge in graph["edges"]
    )


def test_code_graph_does_not_guess_attribute_receiver_or_bare_method_calls(tmp_path):
    _write(
        tmp_path / "model.py",
        "class A:\n"
        "    def save(self):\n"
        "        pass\n\n"
        "class B:\n"
        "    def save(self):\n"
        "        pass\n\n"
        "def caller(obj):\n"
        "    obj.save()\n"
        "    return save()\n",
    )

    graph = build_code_graph(str(tmp_path))

    assert not [edge for edge in graph["edges"] if edge["kind"] == "calls" and edge.get("call") == "save"]


def test_code_graph_preserves_relative_import_marker_and_resolves_alias(tmp_path):
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "sub.py", "def foo():\n    pass\n")
    _write(tmp_path / "pkg" / "main.py", "from .sub import foo\n\ndef caller():\n    return foo()\n")
    _write(tmp_path / "other.py", "def foo():\n    pass\n")

    graph = build_code_graph(str(tmp_path))

    assert any(
        edge["kind"] == "imports"
        and edge["source"] == "pkg/main.py"
        and edge["target"] == ".sub.foo"
        and edge["relative"] is True
        for edge in graph["edges"]
    )
    assert any(
        edge["kind"] == "calls"
        and edge["source"] == "pkg/main.py::caller"
        and edge["target"] == "pkg/sub.py::foo"
        and edge["confidence"] == "import-resolved"
        for edge in graph["edges"]
    )


def test_code_graph_skips_hidden_dirs_symlinks_and_redacts_sensitive_excerpts(tmp_path):
    outside = tmp_path.parent / "outside_codegraph_secret_fixture.py"
    outside.write_text("def outside_symbol():\n    pass\n", encoding="utf-8")
    outside_dir = tmp_path.parent / "outside_codegraph_dir_fixture"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "outside_dir.py").write_text("def outside_dir_symbol():\n    pass\n", encoding="utf-8")
    _write(tmp_path / ".config" / "hidden.py", "def hidden():\n    pass\n")
    (tmp_path / "visible_link.py").symlink_to(outside)
    (tmp_path / "visible_dir_link").symlink_to(outside_dir, target_is_directory=True)
    _write(tmp_path / "visible.py", "def api_key = 'not valid python'\n")
    _write(tmp_path / "dict_secret.py", "CONFIG = {'api_key': 'abc123456789'}\n")
    _write(tmp_path / "main.py", "def safe(secret_access_key='abc123456789'):\n    return secret_access_key\n")

    graph = build_code_graph(str(tmp_path))
    graph_json = json.dumps(graph)

    assert "hidden.py" not in graph_json
    assert "outside_symbol" not in graph_json
    assert "outside_dir_symbol" not in graph_json
    assert "abc123456789" not in graph_json
    assert "not valid python" not in graph_json
    assert graph["skipped_reasons"]["symlink"] >= 2
    assert any(error["file"] == "visible.py" for error in graph["parse_errors"])


def test_code_graph_rejects_overbroad_and_blank_inputs(tmp_path):
    with pytest.raises(ValueError, match="too broad"):
        build_code_graph("/", max_files=1)
    with pytest.raises(ValueError, match="too broad"):
        build_code_graph(str(Path.home()), max_files=1)
    with pytest.raises(ValueError, match="too broad"):
        build_code_graph("/tmp", max_files=1)
    unmarked = tmp_path.parent / f"unmarked-{tmp_path.name}"
    unmarked.mkdir()
    _write(unmarked / "main.py", "def unsafe():\n    pass\n")
    with pytest.raises(ValueError, match="project/corpus root"):
        build_code_graph(str(unmarked), max_files=1)
    with pytest.raises(ValueError, match="non-empty"):
        build_code_graph("   ", max_files=1)
    with pytest.raises(ValueError, match="non-empty"):
        code_graph_context(str(tmp_path), "   ")
    with pytest.raises(ValueError, match="non-empty"):
        code_graph_impact(str(tmp_path), "")


def test_code_graph_reads_python_encoding_cookie_files(tmp_path):
    encoded = "# -*- coding: latin-1 -*-\ndef café():\n    return 'olé'\n".encode("latin-1")
    (tmp_path / "latin1.py").write_bytes(encoded)

    graph = build_code_graph(str(tmp_path))

    assert any(node["id"] == "latin1.py::café" for node in graph["nodes"])
    assert graph["stats"]["parse_errors"] == 0


def test_code_graph_context_includes_class_member_symbols(tmp_path):
    _write(
        tmp_path / "model.py",
        "class Service:\n"
        "    def start(self):\n"
        "        pass\n\n"
        "    def stop(self):\n"
        "        pass\n",
    )

    context = code_graph_context(str(tmp_path), "Service")

    member_ids = {node["id"] for node in context["member_symbols"]}
    assert "model.py::Service.start" in member_ids
    assert "model.py::Service.stop" in member_ids


def test_code_graph_context_and_impact_outputs_stay_bounded(tmp_path):
    _write(tmp_path / "target.py", "def target():\n    pass\n")
    for index in range(250):
        _write(
            tmp_path / f"caller_{index:03}.py",
            "from target import target\n\n"
            f"def caller_{index:03}():\n"
            "    return target()\n",
        )

    context = code_graph_context(str(tmp_path), "target", max_files=500, limit=100)
    impact = code_graph_impact(str(tmp_path), "target", max_files=500, depth=1)

    assert len(context["related_edges"]) == 200
    assert len(impact["impacted"]) == 200
    assert len(impact["impact_edges"]) == 200
    assert all(edge["confidence"] in {"import-resolved", "same-file"} for edge in impact["impact_edges"])


def test_code_graph_exact_max_files_limit_is_not_truncation(tmp_path):
    for index in range(2):
        _write(tmp_path / f"mod_{index}.py", f"def symbol_{index}():\n    pass\n")

    graph = build_code_graph(str(tmp_path), max_files=2)

    assert graph["stats"]["files_seen"] == 2
    assert graph["truncation"]["max_files_reached"] is False
    assert "max_files_reached" not in graph["skipped_reasons"]
    assert not [sample for sample in graph["skipped_samples"] if sample["reason"] == "max_files_reached"]


def test_code_graph_records_when_max_files_limit_truncates_index(tmp_path):
    for index in range(3):
        _write(tmp_path / f"mod_{index}.py", f"def symbol_{index}():\n    pass\n")

    graph = build_code_graph(str(tmp_path), max_files=2)

    assert graph["stats"]["files_seen"] == 2
    assert graph["truncation"]["max_files_reached"] is True
    assert graph["skipped_reasons"]["max_files_reached"] == 1
    assert any(
        sample["reason"] == "max_files_reached" and sample["path"] == "[TRUNCATED]"
        for sample in graph["skipped_samples"]
    )


def test_code_graph_toolset_is_explicit_opt_in_not_default_surface():
    expected = {"code_graph_index", "code_graph_context", "code_graph_impact"}
    names = {tool["function"]["name"] for tool in get_tool_definitions(enabled_toolsets=["code_graph"], quiet_mode=True)}
    assert expected <= names

    for toolset in ["hermes-cli", "hermes-acp", "hermes-api-server", "hermes-telegram"]:
        surface_names = {tool["function"]["name"] for tool in get_tool_definitions(enabled_toolsets=[toolset], quiet_mode=True)}
        assert not expected.intersection(surface_names)


def test_code_graph_toolset_is_discovered_without_prior_module_import():
    script = """
from model_tools import get_tool_definitions
expected = {"code_graph_index", "code_graph_context", "code_graph_impact"}
names = {tool["function"]["name"] for tool in get_tool_definitions(enabled_toolsets=["code_graph"], quiet_mode=True)}
missing = expected - names
assert not missing, missing
"""
    subprocess.run([sys.executable, "-c", script], check=True)
