#!/usr/bin/env python3
"""
Tests for the read_code_structure tool.

Covers:
- Python AST extraction (functions, classes, methods, decorators, docstrings,
  nested structures, async functions, syntax errors)
- Regex-based extraction for JavaScript, TypeScript, Go, Java, C/C++, Ruby,
  Rust, PHP, R
- Handler-level tests (missing path, non-existent file, unsupported extension,
  file-not-found suggestions, empty files)
- Docstring/comment extraction for multiple languages

Run with:  python -m pytest tests/tools/test_read_code_structure.py -v
"""

import json
import os
import shutil
import tempfile
import unittest

from tools.read_code_structure_tool import (
    ALL_CODE_EXTENSIONS,
    READ_CODE_STRUCTURE_SCHEMA,
    _extract_python_structure,
    _extract_regex_structure,
    _find_docstring_above_line,
    _handle_read_code_structure,
    check_code_structure_requirements,
)
from tools.registry import tool_error, tool_result


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------

class TestPythonASTExtraction(unittest.TestCase):
    """Tests for _extract_python_structure using the ast module."""

    def test_simple_function(self):
        source = '''def hello():
    """Say hello."""
    print("hello")
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "hello")
        self.assertEqual(entries[0]["type"], "function")
        self.assertEqual(entries[0]["line"], 1)
        self.assertEqual(entries[0]["end_line"], 3)
        self.assertEqual(entries[0]["docstring"], "Say hello.")
        self.assertIsNone(entries[0]["parent"])

    def test_async_function(self):
        source = '''async def fetch_data(url):
    """Fetch data from URL."""
    pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "fetch_data")
        self.assertEqual(entries[0]["type"], "function")
        self.assertEqual(entries[0]["docstring"], "Fetch data from URL.")

    def test_class_with_methods(self):
        source = '''class Calculator:
    """A simple calculator."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b
'''
        entries = _extract_python_structure(source)
        # 1 class + 2 methods = 3 entries
        self.assertEqual(len(entries), 3)

        # Class entry
        cls = entries[0]
        self.assertEqual(cls["name"], "Calculator")
        self.assertEqual(cls["type"], "class")
        self.assertEqual(cls["docstring"], "A simple calculator.")
        self.assertIsNone(cls["parent"])

        # Methods
        add_method = entries[1]
        self.assertEqual(add_method["name"], "add")
        self.assertEqual(add_method["type"], "method")
        self.assertEqual(add_method["parent"], "Calculator")
        self.assertEqual(add_method["docstring"], "Add two numbers.")

        sub_method = entries[2]
        self.assertEqual(sub_method["name"], "subtract")
        self.assertEqual(sub_method["type"], "method")
        self.assertEqual(sub_method["parent"], "Calculator")

    def test_decorators(self):
        source = '''@staticmethod
@cache
def get_config():
    """Get the config."""
    pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 1)
        self.assertIn("staticmethod", entries[0]["decorators"])
        self.assertIn("cache", entries[0]["decorators"])

    def test_decorator_with_args(self):
        source = '''@route("/api/v1/users")
def list_users():
    """List all users."""
    pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 1)
        self.assertIn("route", entries[0]["decorators"])

    def test_nested_class(self):
        source = '''class Outer:
    """Outer class."""
    class Inner:
        """Inner class."""
        pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 2)
        outer = entries[0]
        inner = entries[1]
        self.assertEqual(outer["name"], "Outer")
        self.assertEqual(inner["name"], "Inner")
        self.assertEqual(inner["parent"], "Outer")

    def test_function_without_docstring(self):
        source = '''def no_doc():
    pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["docstring"])

    def test_class_without_docstring(self):
        source = '''class NoDoc:
    pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["docstring"])

    def test_syntax_error_returns_empty(self):
        source = '''def broken(
    this is not valid python
'''
        entries = _extract_python_structure(source)
        self.assertEqual(entries, [])

    def test_empty_source(self):
        entries = _extract_python_structure("")
        self.assertEqual(entries, [])

    def test_module_level_only(self):
        source = '''import os

CONSTANT = 42

def foo():
    pass

x = 1
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "foo")

    def test_nested_function_in_function(self):
        source = '''def outer():
    """Outer function."""
    def inner():
        """Inner function."""
        pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["name"], "outer")
        self.assertEqual(entries[0]["type"], "function")
        self.assertEqual(entries[1]["name"], "inner")
        self.assertEqual(entries[1]["type"], "function")
        self.assertEqual(entries[1]["parent"], "outer")

    def test_line_numbers_accurate(self):
        source = '''# comment line 1
# comment line 2

def first():
    pass

class MyClass:
    def method(self):
        pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(entries[0]["line"], 4)   # def first
        self.assertEqual(entries[1]["line"], 7)   # class MyClass
        self.assertEqual(entries[2]["line"], 8)   # def method

    def test_multiple_classes_and_functions(self):
        source = '''def standalone():
    pass

class A:
    def a_method(self):
        pass

class B:
    def b_method(self):
        pass

def another_standalone():
    pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(len(entries), 6)
        names = [e["name"] for e in entries]
        self.assertEqual(names, ["standalone", "A", "a_method", "B", "b_method", "another_standalone"])


# ---------------------------------------------------------------------------
# Regex-based extraction — JavaScript
# ---------------------------------------------------------------------------

class TestJavaScriptExtraction(unittest.TestCase):
    def test_function_declaration(self):
        source = '''function hello() {
  return "hello";
}
'''
        entries = _extract_regex_structure(source, ".js")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("hello", func_names)

    def test_async_function(self):
        source = '''async function fetchData() {
  return await fetch(url);
}
'''
        entries = _extract_regex_structure(source, ".js")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("fetchData", func_names)

    def test_export_function(self):
        source = '''export function helper() {}
'''
        entries = _extract_regex_structure(source, ".js")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("helper", func_names)

    def test_arrow_function(self):
        source = '''const add = (a, b) => a + b;
'''
        entries = _extract_regex_structure(source, ".js")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("add", func_names)

    def test_class_declaration(self):
        source = '''class DataProcessor {
  process(input) {
    return input.trim();
  }
}
'''
        entries = _extract_regex_structure(source, ".js")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("DataProcessor", class_names)

    def test_export_class(self):
        source = '''export default class App extends React.Component {}
'''
        entries = _extract_regex_structure(source, ".js")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("App", class_names)


# ---------------------------------------------------------------------------
# Regex-based extraction — TypeScript
# ---------------------------------------------------------------------------

class TestTypeScriptExtraction(unittest.TestCase):
    def test_typed_function(self):
        source = '''function greet(name: string): string {
  return `Hello, ${name}`;
}
'''
        entries = _extract_regex_structure(source, ".ts")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("greet", func_names)

    def test_generic_function(self):
        source = '''function identity<T>(arg: T): T {
  return arg;
}
'''
        entries = _extract_regex_structure(source, ".ts")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("identity", func_names)

    def test_abstract_class(self):
        source = '''export abstract class BaseService {
  abstract execute(): void;
}
'''
        entries = _extract_regex_structure(source, ".ts")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("BaseService", class_names)


# ---------------------------------------------------------------------------
# Regex-based extraction — Go
# ---------------------------------------------------------------------------

class TestGoExtraction(unittest.TestCase):
    def test_package_function(self):
        source = '''package main

func hello() string {
    return "hello"
}
'''
        entries = _extract_regex_structure(source, ".go")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("hello", func_names)

    def test_method_with_receiver(self):
        source = '''func (c *Calculator) Add(a, b int) int {
    return a + b
}
'''
        entries = _extract_regex_structure(source, ".go")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("Add", func_names)

    def test_struct(self):
        source = '''type Server struct {
    Host string
    Port int
}
'''
        entries = _extract_regex_structure(source, ".go")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("Server", class_names)


# ---------------------------------------------------------------------------
# Regex-based extraction — Java
# ---------------------------------------------------------------------------

class TestJavaExtraction(unittest.TestCase):
    def test_class_declaration(self):
        source = '''public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
'''
        entries = _extract_regex_structure(source, ".java")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("HelloWorld", class_names)

    def test_interface(self):
        source = '''public interface Runnable {
    void run();
}
'''
        entries = _extract_regex_structure(source, ".java")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("Runnable", class_names)


# ---------------------------------------------------------------------------
# Regex-based extraction — Ruby
# ---------------------------------------------------------------------------

class TestRubyExtraction(unittest.TestCase):
    def test_method_definition(self):
        source = '''def greet(name)
  puts "Hello, #{name}!"
end
'''
        entries = _extract_regex_structure(source, ".rb")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("greet", func_names)

    def test_self_method(self):
        source = '''def self.create(params)
  new(params)
end
'''
        entries = _extract_regex_structure(source, ".rb")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("create", func_names)

    def test_class(self):
        source = '''class User
  def initialize(name)
    @name = name
  end
end
'''
        entries = _extract_regex_structure(source, ".rb")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("User", class_names)

    def test_module(self):
        source = '''module Authenticatable
  def authenticate
    true
  end
end
'''
        entries = _extract_regex_structure(source, ".rb")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("Authenticatable", class_names)

    def test_predicate_method(self):
        source = '''def active?
  status == "active"
end
'''
        entries = _extract_regex_structure(source, ".rb")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("active?", func_names)

    def test_bang_method(self):
        source = '''def save!
  raise "Invalid" unless valid?
end
'''
        entries = _extract_regex_structure(source, ".rb")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("save!", func_names)


# ---------------------------------------------------------------------------
# Regex-based extraction — Rust
# ---------------------------------------------------------------------------

class TestRustExtraction(unittest.TestCase):
    def test_fn_declaration(self):
        source = '''fn main() {
    println!("Hello");
}
'''
        entries = _extract_regex_structure(source, ".rs")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("main", func_names)

    def test_pub_fn(self):
        source = '''pub fn new(config: Config) -> Self {
    Self { config }
}
'''
        entries = _extract_regex_structure(source, ".rs")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("new", func_names)

    def test_async_fn(self):
        source = '''pub async fn fetch(url: &str) -> Result<String> {
    Ok(reqwest::get(url).await?.text().await?)
}
'''
        entries = _extract_regex_structure(source, ".rs")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("fetch", func_names)

    def test_struct(self):
        source = '''pub struct Server {
    host: String,
    port: u16,
}
'''
        entries = _extract_regex_structure(source, ".rs")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("Server", class_names)

    def test_trait(self):
        source = '''pub trait Handler {
    fn handle(&self, req: Request) -> Response;
}
'''
        entries = _extract_regex_structure(source, ".rs")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("Handler", class_names)

    def test_enum(self):
        source = '''pub enum Color {
    Red,
    Green,
    Blue,
}
'''
        entries = _extract_regex_structure(source, ".rs")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("Color", class_names)


# ---------------------------------------------------------------------------
# Regex-based extraction — PHP
# ---------------------------------------------------------------------------

class TestPHPExtraction(unittest.TestCase):
    def test_function(self):
        source = '''<?php
function greet($name) {
    return "Hello, " . $name;
}
'''
        entries = _extract_regex_structure(source, ".php")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("greet", func_names)

    def test_class(self):
        source = '''<?php
class User {
    public function getName() {
        return $this->name;
    }
}
'''
        entries = _extract_regex_structure(source, ".php")
        class_names = [e["name"] for e in entries if e["type"] == "class"]
        self.assertIn("User", class_names)


# ---------------------------------------------------------------------------
# Regex-based extraction — R
# ---------------------------------------------------------------------------

class TestRExtraction(unittest.TestCase):
    def test_function_assignment(self):
        source = '''add <- function(a, b) {
  a + b
}
'''
        entries = _extract_regex_structure(source, ".R")
        func_names = [e["name"] for e in entries if e["type"] == "function"]
        self.assertIn("add", func_names)


# ---------------------------------------------------------------------------
# Docstring / comment extraction
# ---------------------------------------------------------------------------

class TestDocstringExtraction(unittest.TestCase):
    def test_python_docstring_via_ast(self):
        source = '''def foo():
    """This is a docstring."""
    pass
'''
        entries = _extract_python_structure(source)
        self.assertEqual(entries[0]["docstring"], "This is a docstring.")

    def test_js_doc_comment_above_function(self):
        source = '''/**
 * Calculate the sum.
 * @param a first number
 */
function sum(a, b) {
  return a + b;
}
'''
        doc = _find_docstring_above_line(source, 5, ".js")
        self.assertIsNotNone(doc)
        self.assertIn("Calculate the sum", doc)

    def test_go_comment_above_func(self):
        source = '''// Add returns the sum of a and b.
func Add(a, b int) int {
    return a + b
}
'''
        doc = _find_docstring_above_line(source, 2, ".go")
        self.assertIsNotNone(doc)
        self.assertIn("Add returns the sum", doc)

    def test_rust_doc_comment(self):
        source = '''/// Fetches data from the given URL.
/// Returns a Result on success.
pub async fn fetch(url: &str) -> Result<String> {
    todo!()
}
'''
        doc = _find_docstring_above_line(source, 3, ".rs")
        self.assertIsNotNone(doc)
        self.assertIn("Fetches data", doc)

    def test_no_comment_returns_none(self):
        source = '''function noComment() {}
'''
        doc = _find_docstring_above_line(source, 1, ".js")
        self.assertIsNone(doc)

    def test_python_returns_none(self):
        # Python docstrings are handled by AST, not regex
        doc = _find_docstring_above_line("def foo():\n    pass\n", 1, ".py")
        self.assertIsNone(doc)


# ---------------------------------------------------------------------------
# Handler-level tests
# ---------------------------------------------------------------------------

class TestHandler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rcs_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_missing_path_returns_error(self):
        result = _handle_read_code_structure({})
        self.assertIn("error", result)

    def test_empty_path_returns_error(self):
        result = _handle_read_code_structure({"path": ""})
        self.assertIn("error", result)

    def test_nonexistent_file_returns_error(self):
        result = _handle_read_code_structure({"path": "/nonexistent/file.py"})
        self.assertIn("error", result)

    def test_unsupported_extension_returns_error(self):
        path = self._write("data.txt", "hello world")
        result = _handle_read_code_structure({"path": path})
        self.assertIn("error", result)
        self.assertIn("Unsupported", result)

    def test_python_file_returns_structure(self):
        path = self._write("example.py", '''"""Module docstring."""

def hello():
    """Say hello."""
    print("hello")

class Greeter:
    """A greeting class."""

    def greet(self, name):
        """Greet someone."""
        return f"Hello, {name}"
''')
        result = _handle_read_code_structure({"path": path})
        # Result should be parseable JSON
        data = json.loads(result)
        self.assertIn("entries", data)
        self.assertEqual(data["language"], "py")
        # 1 function + 1 class + 1 method = 3 entries
        self.assertEqual(len(data["entries"]), 3)

    def test_go_file_returns_structure(self):
        path = self._write("main.go", '''package main

import "fmt"

// Calculator performs math
type Calculator struct {
    name string
}

// Add adds two numbers
func (c *Calculator) Add(a, b int) int {
    return a + b
}

func main() {
    fmt.Println("hello")
}
''')
        result = _handle_read_code_structure({"path": path})
        data = json.loads(result)
        self.assertEqual(data["language"], "go")
        self.assertTrue(len(data["entries"]) >= 2)  # At least Add and Calculator

    def test_js_file_returns_structure(self):
        path = self._write("app.js", '''class App {
  constructor() {
    this.name = "app";
  }
}

async function run() {
  return Promise.resolve(42);
}
''')
        result = _handle_read_code_structure({"path": path})
        data = json.loads(result)
        self.assertEqual(data["language"], "js")
        self.assertTrue(len(data["entries"]) >= 1)

    def test_file_with_no_structure(self):
        path = self._write("empty.py", "# Just a comment\nx = 1\n")
        result = _handle_read_code_structure({"path": path})
        data = json.loads(result)
        self.assertEqual(data["entries"], [])

    def test_tilde_expansion(self):
        # Write a file using ~/tmp path — only works if ~/tmp exists
        home_tmp = os.path.expanduser("~/tmp_rcs_test")
        try:
            os.makedirs(home_tmp, exist_ok=True)
            fpath = os.path.join(home_tmp, "test_tilde.py")
            with open(fpath, "w") as f:
                f.write("def foo(): pass\n")
            result = _handle_read_code_structure({"path": "~/tmp_rcs_test/test_tilde.py"})
            data = json.loads(result)
            self.assertIn("entries", data)
        finally:
            shutil.rmtree(home_tmp, ignore_errors=True)

    def test_file_not_found_suggests_similar(self):
        # Create a file, then ask for a similar name
        self._write("calculator.py", "def add(): pass\n")
        path = os.path.join(self.tmp, "calculatr.py")
        result = _handle_read_code_structure({"path": path})
        # Should suggest "calculator.py"
        self.assertIn("Did you mean", result)

    def test_rust_file_returns_structure(self):
        path = self._write("lib.rs", '''pub struct Config {
    pub host: String,
}

pub fn load() -> Config {
    Config { host: "localhost".into() }
}
''')
        result = _handle_read_code_structure({"path": path})
        data = json.loads(result)
        self.assertEqual(data["language"], "rs")
        self.assertTrue(len(data["entries"]) >= 2)


# ---------------------------------------------------------------------------
# Schema and registry
# ---------------------------------------------------------------------------

class TestSchemaAndRegistry(unittest.TestCase):
    def test_schema_has_required_fields(self):
        self.assertEqual(READ_CODE_STRUCTURE_SCHEMA["name"], "read_code_structure")
        self.assertIn("parameters", READ_CODE_STRUCTURE_SCHEMA)
        self.assertIn("path", READ_CODE_STRUCTURE_SCHEMA["parameters"]["properties"])
        self.assertIn("path", READ_CODE_STRUCTURE_SCHEMA["parameters"]["required"])

    def test_schema_description_mentions_languages(self):
        desc = READ_CODE_STRUCTURE_SCHEMA["description"]
        self.assertIn("Python", desc)
        self.assertIn("JavaScript", desc)
        self.assertIn("Go", desc)
        self.assertIn("Rust", desc)

    def test_all_extensions_are_covered(self):
        expected = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java",
                    ".c", ".cpp", ".h", ".hpp", ".rb", ".rs", ".php", ".r", ".R"}
        self.assertEqual(ALL_CODE_EXTENSIONS, expected)

    def test_check_requirements_always_true(self):
        self.assertTrue(check_code_structure_requirements())


# ---------------------------------------------------------------------------
# Regex entry line-number accuracy
# ---------------------------------------------------------------------------

class TestRegexLineNumbers(unittest.TestCase):
    def test_go_line_numbers(self):
        source = '''package main

func first() {}
func second() {}
func third() {}
'''
        entries = _extract_regex_structure(source, ".go")
        func_entries = [e for e in entries if e["type"] == "function"]
        # Sorted by line number
        lines = [e["line"] for e in func_entries]
        self.assertEqual(lines, sorted(lines))
        self.assertTrue(all(l > 0 for l in lines))

    def test_js_class_before_function(self):
        source = '''class App {}

function init() {}
'''
        entries = _extract_regex_structure(source, ".js")
        # Class should come before function in line order
        class_entry = next(e for e in entries if e["type"] == "class")
        func_entry = next(e for e in entries if e["type"] == "function")
        self.assertLess(class_entry["line"], func_entry["line"])


if __name__ == "__main__":
    unittest.main()
