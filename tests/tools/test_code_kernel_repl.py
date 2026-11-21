#!/usr/bin/env python3
"""REPL contract for execute_code: echo of the last expression.

Persistent kernels print the repr of a final expression statement
followed by a newline, after any prior prints. Assignments, imports,
def/class, None results, empty cells and comment-only cells produce
no echo. A print() call never appends an extra None line.
"""

import json
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")


from tools.code_execution_tool import execute_code
from tools.code_kernel import shutdown_all_kernels


@contextmanager
def _kernel_config(**overrides):
    config = {"mode": "strict", "timeout": 30}
    config.update(overrides)
    with patch("tools.code_execution_tool._load_config", return_value=config):
        yield


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def _run(code, **kwargs):
    kwargs.setdefault("task_id", "kernel-repl-test")
    return json.loads(execute_code(code, **kwargs))


class TestReplLastExpressionEcho(unittest.TestCase):
    def test_int_expression_echo_after_assignment(self):
        with _kernel_config():
            first = _run("repl_a = 40")
            self.assertEqual(first["status"], "success", first)
            self.assertEqual(first["output"], "")
            second = _run("repl_a + 2")
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(second["output"], "42\n")

    def test_string_and_list_repr_are_exact(self):
        with _kernel_config():
            s = _run("'hola-repl'")
            self.assertEqual(s["status"], "success", s)
            self.assertEqual(s["output"], "'hola-repl'\n")
            lst = _run("[1, 2, 3]")
        self.assertEqual(lst["status"], "success", lst)
        self.assertEqual(lst["output"], "[1, 2, 3]\n")

    def test_none_and_none_returning_call_have_no_echo(self):
        with _kernel_config():
            none_lit = _run("None")
            self.assertEqual(none_lit["status"], "success", none_lit)
            self.assertEqual(none_lit["output"], "")
            _run("def _repl_none_fn():\n    return None")
            call_none = _run("_repl_none_fn()")
        self.assertEqual(call_none["status"], "success", call_none)
        self.assertEqual(call_none["output"], "")

    def test_print_does_not_add_extra_none_line(self):
        with _kernel_config():
            res = _run("print('antes-repl')")
        self.assertEqual(res["status"], "success", res)
        self.assertEqual(res["output"], "antes-repl\n")

    def test_print_then_expr_keeps_exact_order(self):
        with _kernel_config():
            res = _run("print('before-repl')\n2 + 3")
        self.assertEqual(res["status"], "success", res)
        self.assertEqual(res["output"], "before-repl\n5\n")

    def test_assignment_import_def_class_have_no_echo(self):
        with _kernel_config():
            self.assertEqual(_run("repl_b = 7")["output"], "")
            self.assertEqual(_run("import math as _repl_math")["output"], "")
            self.assertEqual(_run("def _repl_fn():\n    return 1")["output"], "")
            self.assertEqual(_run("class _ReplKls:\n    pass")["output"], "")

    def test_comments_only_cell_has_no_echo(self):
        with _kernel_config():
            res = _run("# solo comentario repl\n# otra linea\n")
        self.assertEqual(res["status"], "success", res)
        self.assertEqual(res["output"], "")

    def test_mutating_call_evaluated_exactly_once(self):
        with _kernel_config():
            _run(
                "repl_counter = 0\n"
                "def _repl_inc():\n"
                "    global repl_counter\n"
                "    repl_counter += 1\n"
                "    return repl_counter"
            )
            res = _run("_repl_inc()")
            self.assertEqual(res["status"], "success", res)
            self.assertEqual(res["output"], "1\n")
            check = _run("repl_counter")
        self.assertEqual(check["status"], "success", check)
        self.assertEqual(check["output"], "1\n")

    def test_syntax_error_runs_no_prefix_side_effect(self):
        with _kernel_config():
            res = _run("repl_prefix_guard = 123\nreturn 1")
            self.assertEqual(res["status"], "error", res)
            self.assertIn("SyntaxError", res["error"], res)
            probe = _run("print(repl_prefix_guard)")
        self.assertEqual(probe["status"], "error", probe)
        self.assertIn("NameError", probe["error"], probe)

    def test_final_expression_compile_error_runs_no_prefix_side_effect(self):
        # yield/await parse fine and only fail at compile time: echoing the final expression
        # must not run the statements before it when that expression cannot compile.
        with _kernel_config():
            for tail in ("yield 1", "await repl_tail_guard"):
                res = _run(f"repl_tail_guard = 123\n{tail}")
                self.assertEqual(res["status"], "error", res)
                self.assertIn("SyntaxError", res["error"], res)
                probe = _run("print(repl_tail_guard)")
                self.assertEqual(probe["status"], "error", probe)
                self.assertIn("NameError", probe["error"], probe)

    def test_zero_division_keeps_prior_stdout_and_kernel_alive(self):
        with _kernel_config():
            res = _run("print('prev-repl')\n1 / 0")
            self.assertEqual(res["status"], "error", res)
            self.assertIn("ZeroDivisionError", res["error"], res)
            self.assertTrue(
                res["output"].startswith("prev-repl\n"),
                res,
            )
            _run("repl_alive = 77")
            alive = _run("repl_alive + 1")
        self.assertEqual(alive["status"], "success", alive)
        self.assertEqual(alive["output"], "78\n")

    def test_repr_error_is_normal_cell_error(self):
        with _kernel_config():
            _run(
                "class _ReplBadRepr:\n"
                "    def __repr__(self):\n"
                "        raise RuntimeError('boom-repr')\n"
                "repl_bad = _ReplBadRepr()"
            )
            res = _run("repl_bad")
            self.assertEqual(res["status"], "error", res)
            self.assertIn("boom-repr", res["error"] + res["output"], res)
            _run("repl_after_repr = 5")
            alive = _run("repl_after_repr + 1")
        self.assertEqual(alive["status"], "success", alive)
        self.assertEqual(alive["output"], "6\n")

    def test_user_names_do_not_collide_with_implementation(self):
        with _kernel_config():
            _run("_repl_tmp_xyz = 10\n_result = 1\n_expr = 2")
            res = _run("_repl_tmp_xyz + 5")
            self.assertEqual(res["status"], "success", res)
            self.assertEqual(res["output"], "15\n")
            still = _run("_result + _expr")
        self.assertEqual(still["status"], "success", still)
        self.assertEqual(still["output"], "3\n")

    def test_unicode_repr_passes_through(self):
        with _kernel_config():
            res = _run("'ñé✅' * 3")
        self.assertEqual(res["status"], "success", res)
        self.assertEqual(res["output"], "'ñé✅ñé✅ñé✅'\n")

    def test_large_repr_does_not_break_protocol(self):
        with _kernel_config():
            res = _run("list(range(20000))")
        self.assertEqual(res["status"], "success", res)
        self.assertIn("0", res["output"], res)
        self.assertIn("19999", res["output"], res)
        alive = _run("1 + 1")
        self.assertEqual(alive["status"], "success", alive)
        self.assertEqual(alive["output"], "2\n")

    def test_future_import_then_expr(self):
        with _kernel_config():
            _run("from __future__ import annotations\nrepl_ann: int = 5")
            res = _run("repl_ann + 1")
        self.assertEqual(res["status"], "success", res)
        self.assertEqual(res["output"], "6\n")


if __name__ == "__main__":
    unittest.main()
