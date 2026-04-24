"""Tests for tools/lsp_rename_tool.py."""

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

from tools import file_state
from tools.file_tools import _file_ops_cache, _file_ops_lock, _read_tracker, read_file_tool
from tools.registry import registry


def _write_python_file(tmp_path, content: str, name: str = "sample.py"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestLspRenameRegistration:
    def test_registry_entry_and_schema(self):
        from tools.lsp_rename_tool import LSP_RENAME_SCHEMA

        entry = registry.get_entry("lsp_rename")
        assert entry is not None
        assert entry.toolset == "code_intel"
        assert entry.schema == LSP_RENAME_SCHEMA

        params = entry.schema["parameters"]
        assert set(params["required"]) == {"path", "line", "column", "new_name"}
        assert params["properties"]["apply"]["default"] is False
        assert params["properties"]["max_files"]["default"] == 1


class TestLspRenameValidation:
    def test_invalid_identifier_returns_structured_error(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo():\n    value = 1\n    return value\n",
        )

        result = json.loads(
            lsp_rename_tool(
                path=str(path),
                line=2,
                column=4,
                new_name="123bad",
            )
        )

        assert result["error"] == "Invalid Python identifier for rename target."
        assert result["code"] == "invalid_new_name"

    def test_unsupported_language_returns_structured_error(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(tmp_path, "const value = 1;\n", name="sample.js")

        result = json.loads(
            lsp_rename_tool(
                path=str(path),
                line=1,
                column=6,
                new_name="renamed",
            )
        )

        assert result["error"] == "Unsupported language for lsp_rename."
        assert result["code"] == "unsupported_language"
        assert result["supported_languages"] == ["python"]

    def test_max_files_gt_one_still_allows_bounded_single_file_rename(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo():\n    value = 1\n    return value\n",
        )

        result = json.loads(
            lsp_rename_tool(
                path=str(path),
                line=2,
                column=4,
                new_name="count",
                max_files=2,
                apply=False,
            )
        )

        assert result["success"] is True
        assert result["changed_files"] == 1
        assert result["max_files"] == 2


class TestLspRenamePythonFallback:
    def teardown_method(self):
        _read_tracker.clear()
        file_state.get_registry().clear()

    def test_python_local_variable_preview_returns_diff_without_mutation(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        original = (
            "def demo():\n"
            "    value = 1\n"
            "    total = value + 2\n"
            "    return total\n"
        )
        path = _write_python_file(tmp_path, original)

        result = json.loads(
            lsp_rename_tool(
                path=str(path),
                line=2,
                column=4,
                new_name="count",
                apply=False,
            )
        )

        assert result["success"] is True
        assert result["applied"] is False
        assert result["engine"] == "python_ast_single_file"
        assert result["changed_files"] == 1
        assert "-    value = 1" in result["diff"]
        assert "+    count = 1" in result["diff"]
        assert path.read_text(encoding="utf-8") == original

    def test_project_wide_preview_renames_exported_symbols_and_updates_importers(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        project_root = tmp_path / "project"
        pkg = project_root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")

        cases = [
            (
                "def greet():\n    return 'hi'\n",
                "from pkg.a import greet\n\nRESULT = greet()\n",
                1,
                4,
                "welcome",
                [
                    "-def greet():",
                    "+def welcome():",
                    "-from pkg.a import greet",
                    "+from pkg.a import welcome",
                    "-RESULT = greet()",
                    "+RESULT = welcome()",
                ],
            ),
            (
                "class Greeter:\n    pass\n",
                "from pkg.a import Greeter\n\nRESULT = Greeter()\n",
                1,
                6,
                "Messenger",
                [
                    "-class Greeter:",
                    "+class Messenger:",
                    "-from pkg.a import Greeter",
                    "+from pkg.a import Messenger",
                    "-RESULT = Greeter()",
                    "+RESULT = Messenger()",
                ],
            ),
        ]

        for source_a, source_b, line, column, new_name, diff_expectations in cases:
            (pkg / "a.py").write_text(source_a, encoding="utf-8")
            (pkg / "b.py").write_text(source_b, encoding="utf-8")

            result = json.loads(
                lsp_rename_tool(
                    path=str(pkg / "a.py"),
                    line=line,
                    column=column,
                    new_name=new_name,
                    project_root=str(project_root),
                    max_files=2,
                    apply=False,
                )
            )

            assert result["success"] is True
            assert result["applied"] is False
            assert result["engine"] == "python_ast_project_wide"
            assert result["changed_files"] == 2
            assert result["files"] == [str(pkg / "a.py"), str(pkg / "b.py")]
            for expected in diff_expectations:
                assert expected in result["diff"]


    def test_registry_dispatch_defaults_to_preview_without_mutation(self, tmp_path):
        import tools.lsp_rename_tool  # noqa: F401 - ensure self-registering tool is imported

        original = "def demo():\n    value = 1\n    return value\n"
        path = _write_python_file(tmp_path, original)

        result = json.loads(
            registry.dispatch(
                "lsp_rename",
                {"path": str(path), "line": 2, "column": 4, "new_name": "count"},
                task_id="rename-preview-default",
            )
        )

        assert result["success"] is True
        assert result["applied"] is False
        assert "count = 1" in result["diff"]
        assert path.read_text(encoding="utf-8") == original

    def test_project_wide_rename_refuses_import_aliases(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        project_root = tmp_path / "project"
        pkg = project_root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "a.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
        (pkg / "b.py").write_text("from pkg.a import greet as run_greet\n\nRESULT = run_greet()\n", encoding="utf-8")

        result = json.loads(
            lsp_rename_tool(
                path=str(pkg / "a.py"),
                line=1,
                column=4,
                new_name="welcome",
                project_root=str(project_root),
                max_files=2,
                apply=False,
            )
        )

        assert result["error"] == "Refusing ambiguous project-wide rename."
        assert result["code"] == "ambiguous_workspace_rename"
        assert result["reason"] == "import_alias_not_supported"
        assert result["path"] == str(pkg / "b.py")


    def test_project_wide_rename_refuses_target_module_new_name_collision(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        project_root = tmp_path / "project"
        pkg = project_root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        target = pkg / "a.py"
        target.write_text(
            "def greet():\n    return 'hi'\n\ndef welcome():\n    return 'existing'\n",
            encoding="utf-8",
        )
        (pkg / "b.py").write_text("from pkg.a import greet\n\nRESULT = greet()\n", encoding="utf-8")

        result = json.loads(
            lsp_rename_tool(
                path=str(target),
                line=1,
                column=4,
                new_name="welcome",
                project_root=str(project_root),
                max_files=2,
                apply=False,
            )
        )

        assert result["error"] == "Refusing ambiguous project-wide rename."
        assert result["code"] == "ambiguous_workspace_rename"
        assert result["reason"] == "binding_collision"
        assert result["path"] == str(target)

    def test_project_wide_rename_refuses_importer_new_name_collision(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        project_root = tmp_path / "project"
        pkg = project_root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        target = pkg / "a.py"
        target.write_text("def greet():\n    return 1\n", encoding="utf-8")
        (pkg / "b.py").write_text(
            "from pkg.a import greet\n\ndef local():\n    welcome = 3\n    return greet() + welcome\n",
            encoding="utf-8",
        )

        result = json.loads(
            lsp_rename_tool(
                path=str(target),
                line=1,
                column=4,
                new_name="welcome",
                project_root=str(project_root),
                max_files=2,
                apply=False,
            )
        )

        assert result["error"] == "Refusing ambiguous project-wide rename."
        assert result["code"] == "ambiguous_workspace_rename"
        assert result["reason"] == "binding_collision"
        assert result["path"] == str(pkg / "b.py")


    def test_project_wide_rename_refuses_unproven_qualified_import_consumer(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        project_root = tmp_path / "project"
        pkg = project_root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        target = pkg / "a.py"
        target.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
        (pkg / "b.py").write_text("import pkg.a\n\nRESULT = pkg.a.greet()\n", encoding="utf-8")

        result = json.loads(
            lsp_rename_tool(
                path=str(target),
                line=1,
                column=4,
                new_name="welcome",
                project_root=str(project_root),
                max_files=2,
                apply=True,
            )
        )

        assert result["error"] == "Refusing ambiguous project-wide rename."
        assert result["code"] == "ambiguous_workspace_rename"
        assert result["reason"] == "unsupported_reference_pattern"
        assert result["path"] == str(pkg / "b.py")
        assert target.read_text(encoding="utf-8") == "def greet():\n    return 'hi'\n"

    def test_project_wide_rename_refuses_unproven_reexport_consumer(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        project_root = tmp_path / "project"
        pkg = project_root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "a.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
        (pkg / "__init__.py").write_text("from pkg.a import greet\n", encoding="utf-8")
        (pkg / "b.py").write_text("from pkg import greet\n\nRESULT = greet()\n", encoding="utf-8")

        result = json.loads(
            lsp_rename_tool(
                path=str(pkg / "a.py"),
                line=1,
                column=4,
                new_name="welcome",
                project_root=str(project_root),
                max_files=3,
                apply=False,
            )
        )

        assert result["error"] == "Refusing ambiguous project-wide rename."
        assert result["code"] == "ambiguous_workspace_rename"
        assert result["reason"] == "unsupported_reference_pattern"
        assert result["path"] == str(pkg / "b.py")

    def test_project_wide_apply_aborts_when_any_target_file_is_stale(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        project_root = tmp_path / "project"
        pkg = project_root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        target = pkg / "a.py"
        consumer = pkg / "b.py"
        target.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
        consumer.write_text("from pkg.a import greet\n\nRESULT = greet()\n", encoding="utf-8")

        task_id = "rename-project-strict"
        json.loads(read_file_tool(str(target), task_id=task_id))
        json.loads(read_file_tool(str(consumer), task_id=task_id))
        time.sleep(0.05)
        consumer.write_text("from pkg.a import greet\n\nRESULT = greet()\nMARKER = 'stale'\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            result = json.loads(
                lsp_rename_tool(
                    path=str(target),
                    line=1,
                    column=4,
                    new_name="welcome",
                    project_root=str(project_root),
                    max_files=2,
                    apply=True,
                    task_id=task_id,
                )
            )

        assert result["error"]
        assert result["code"] == "stale_edit_blocked"
        assert result["path"] == str(consumer)
        assert "modified since you last read" in result["error"]
        assert target.read_text(encoding="utf-8") == "def greet():\n    return 'hi'\n"
        assert "MARKER = 'stale'" in consumer.read_text(encoding="utf-8")

    def test_python_local_variable_apply_mutates_file(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo():\n    value = 1\n    return value\n",
        )

        result = json.loads(
            lsp_rename_tool(
                path=str(path),
                line=2,
                column=4,
                new_name="count",
                apply=True,
            )
        )

        assert result["success"] is True
        assert result["applied"] is True
        assert result["changed_files"] == 1
        assert "count = 1" in path.read_text(encoding="utf-8")
        assert "return count" in path.read_text(encoding="utf-8")

    def test_attribute_rename_is_refused_as_unsafe(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo(obj):\n    return obj.value\n",
        )

        result = json.loads(
            lsp_rename_tool(
                path=str(path),
                line=2,
                column=15,
                new_name="count",
                apply=False,
            )
        )

        assert result["error"] == "Refusing unsafe rename target."
        assert result["code"] == "unsafe_target"
        assert result["reason"] == "attribute_access"

    def test_rename_to_existing_local_binding_is_refused(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo():\n    x = 1\n    y = 2\n    return x + y\n",
        )

        result = json.loads(lsp_rename_tool(path=str(path), line=2, column=4, new_name="y", apply=False))

        assert result["error"] == "Refusing unsafe rename target."
        assert result["code"] == "unsafe_target"
        assert result["reason"] == "binding_collision"

    def test_comprehension_scope_shadowing_is_refused(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo():\n    x = 1\n    xs = [x for x in range(3)]\n    return x, xs\n",
        )

        result = json.loads(lsp_rename_tool(path=str(path), line=2, column=4, new_name="y", apply=False))

        assert result["error"] == "Refusing unsafe rename target."
        assert result["code"] == "unsafe_target"
        assert result["reason"] == "comprehension_scope_not_supported"

    def test_nested_scope_capture_is_refused_instead_of_partially_rewritten(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def outer():\n    x = 1\n    def inner():\n        return x\n    return inner()\n",
        )

        result = json.loads(lsp_rename_tool(path=str(path), line=2, column=4, new_name="y", apply=False))

        assert result["error"] == "Refusing unsafe rename target."
        assert result["code"] == "unsafe_target"
        assert result["reason"] == "nested_scope_reference_not_supported"

    def test_project_root_containment_is_enforced(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        outside = _write_python_file(
            tmp_path,
            "def demo():\n    value = 1\n    return value\n",
        )
        project_root = tmp_path / "project"
        project_root.mkdir()

        result = json.loads(
            lsp_rename_tool(
                path=str(outside),
                line=2,
                column=4,
                new_name="count",
                project_root=str(project_root),
                apply=False,
            )
        )

        assert result["error"] == "Path is outside project_root."
        assert result["code"] == "path_outside_project_root"

    def test_signature_annotation_reference_is_refused(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "value = int\n\ndef demo(value: value) -> value:\n    return value\n",
        )

        result = json.loads(lsp_rename_tool(path=str(path), line=3, column=9, new_name="count", apply=False))

        assert result["error"] == "Refusing unsafe rename target."
        assert result["code"] == "unsafe_target"
        assert result["reason"] == "signature_expression_reference_not_supported"

    def test_signature_default_reference_is_refused(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "value = 1\n\ndef demo(value=value):\n    return value\n",
        )

        result = json.loads(lsp_rename_tool(path=str(path), line=3, column=9, new_name="count", apply=False))

        assert result["error"] == "Refusing unsafe rename target."
        assert result["code"] == "unsafe_target"
        assert result["reason"] == "signature_expression_reference_not_supported"

    def test_relative_path_resolves_against_task_live_cwd(self, tmp_path):
        from tools import file_tools
        from tools.lsp_rename_tool import lsp_rename_tool

        start_dir = tmp_path / "start"
        live_dir = tmp_path / "live"
        start_dir.mkdir()
        live_dir.mkdir()
        start_file = _write_python_file(start_dir, "def demo():\n    value = 1\n    return value\n")
        live_file = _write_python_file(live_dir, "def demo():\n    value = 1\n    return value\n")

        fake_ops = SimpleNamespace(env=SimpleNamespace(cwd=str(live_dir)), cwd=str(start_dir))
        with _file_ops_lock:
            previous = _file_ops_cache.get("rename-live-cwd")
            _file_ops_cache["rename-live-cwd"] = fake_ops

        try:
            with patch.dict(os.environ, {"TERMINAL_CWD": str(start_dir)}, clear=False):
                result = json.loads(
                    lsp_rename_tool(
                        path="sample.py",
                        line=2,
                        column=4,
                        new_name="count",
                        apply=True,
                        task_id="rename-live-cwd",
                    )
                )
        finally:
            with file_tools._file_ops_lock:
                if previous is None:
                    file_tools._file_ops_cache.pop("rename-live-cwd", None)
                else:
                    file_tools._file_ops_cache["rename-live-cwd"] = previous

        assert result["success"] is True
        assert live_file.read_text(encoding="utf-8").count("count") == 2
        assert start_file.read_text(encoding="utf-8").count("value") == 2

    def test_registry_dispatch_forwards_task_id_for_relative_path_resolution(self, tmp_path):
        from tools import file_tools
        import tools.lsp_rename_tool  # noqa: F401 - ensure self-registering tool is imported

        start_dir = tmp_path / "start"
        live_dir = tmp_path / "live"
        start_dir.mkdir()
        live_dir.mkdir()
        start_file = _write_python_file(start_dir, "def demo():\n    value = 1\n    return value\n")
        live_file = _write_python_file(live_dir, "def demo():\n    value = 1\n    return value\n")

        fake_ops = SimpleNamespace(env=SimpleNamespace(cwd=str(live_dir)), cwd=str(start_dir))
        with _file_ops_lock:
            previous = _file_ops_cache.get("rename-dispatch-live-cwd")
            _file_ops_cache["rename-dispatch-live-cwd"] = fake_ops

        try:
            with patch.dict(os.environ, {"TERMINAL_CWD": str(start_dir)}, clear=False):
                result = json.loads(registry.dispatch(
                    "lsp_rename",
                    {"path": "sample.py", "line": 2, "column": 4, "new_name": "count", "apply": True},
                    task_id="rename-dispatch-live-cwd",
                ))
        finally:
            with file_tools._file_ops_lock:
                if previous is None:
                    file_tools._file_ops_cache.pop("rename-dispatch-live-cwd", None)
                else:
                    file_tools._file_ops_cache["rename-dispatch-live-cwd"] = previous

        assert result["success"] is True
        assert live_file.read_text(encoding="utf-8").count("count") == 2
        assert start_file.read_text(encoding="utf-8").count("value") == 2

    def test_apply_warns_when_file_became_stale_since_last_read(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo():\n    value = 1\n    return value\n",
        )

        json.loads(read_file_tool(str(path), task_id="rename-warn"))
        time.sleep(0.05)
        path.write_text("def demo():\n    value = 2\n    return value\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "warn"}, clear=False):
            result = json.loads(
                lsp_rename_tool(
                    path=str(path),
                    line=2,
                    column=4,
                    new_name="count",
                    apply=True,
                    task_id="rename-warn",
                )
            )

        assert result["success"] is True
        assert "_warning" in result
        assert "modified since you last read" in result["_warning"]
        assert "count = 2" in path.read_text(encoding="utf-8")

    def test_apply_strict_mode_refuses_stale_write(self, tmp_path):
        from tools.lsp_rename_tool import lsp_rename_tool

        path = _write_python_file(
            tmp_path,
            "def demo():\n    value = 1\n    return value\n",
        )

        json.loads(read_file_tool(str(path), task_id="rename-strict"))
        time.sleep(0.05)
        path.write_text("def demo():\n    value = 3\n    return value\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            result = json.loads(
                lsp_rename_tool(
                    path=str(path),
                    line=2,
                    column=4,
                    new_name="count",
                    apply=True,
                    task_id="rename-strict",
                )
            )

        assert result["error"]
        assert result["code"] == "stale_edit_blocked"
        assert "modified since you last read" in result["error"]
