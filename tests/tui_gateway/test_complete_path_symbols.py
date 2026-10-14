"""`@symbol:` completion: the per-language definition index behind `complete.path`.

The index is advertised for every extension in ``_SYMBOL_EXTS``; each language fixture pins that its
ordinary function/method/type definitions are found AND that call sites, control flow and prototypes
on the same shapes are not (a C-family ``type name(`` pattern is easy to over-match).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tui_gateway import server

# (filename, source, definitions that must be indexed, names that must NOT be)
_FIXTURES = [
    ("mod.py", "async def fetch_rows(db):\n    return helper(db)\nclass Loader:\n    def run(self): ...\n",
     {"fetch_rows", "Loader", "run"}, {"helper"}),
    ("app.ts", "export async function loadUser(id: string) {}\nexport const useThing = (a: number): number => a\n"
               "export interface Props {}\nexport default class Widget {}\nawait loadUser('x')\n",
     {"loadUser", "useThing", "Props", "Widget"}, set()),
    ("svc.go", "func (s *Server) Handle(w http.ResponseWriter) {\n\tgo s.Run()\n}\ntype Server struct {}\n",
     {"Handle", "Server"}, {"Run"}),
    ("lib.rs", "pub(crate) async fn spawn_task() {}\npub struct Pool;\n", {"spawn_task", "Pool"}, set()),
    ("util.c",
     "static int parse_header(const char *buf, size_t n) {\n    if (n == 0) {\n        return decode(buf);\n    }\n"
     "    else if (buf) {\n        free(buf);\n    }\n}\nint prototype_only(int x);\nstruct packet {\n};\n"
     "typedef enum color { RED } color_t;\n",
     {"parse_header", "packet", "color"}, {"decode", "free", "if", "prototype_only"}),
    ("vec.cpp",
     "template <typename T>\nstd::vector<T> Matrix::row(size_t i) const {\n    return data_.at(i);\n}\n"
     "Matrix::Matrix(int n) : n_(n) {\n}\nclass Matrix : public Base {\n    virtual ~Matrix();\n};\n"
     "const std::string& name() const {\n    delete ptr(1);\n}\n",
     {"row", "Matrix", "name"}, {"at", "ptr"}),
    ("Repo.java",
     "public class Repo {\n    @Override\n    public List<User> findAll(int limit) {\n        return query(limit);\n    }\n"
     "    void helper() {\n        synchronized (lock) {\n            throw new IllegalStateException(\"x\");\n        }\n    }\n"
     "    public Repo(Db db) {\n        this.db = db;\n    }\n}\n",
     {"Repo", "findAll", "helper"}, {"query", "IllegalStateException", "synchronized"}),
    ("Service.cs",
     "public sealed class Service : IService {\n    public async Task<int> RunAsync(CancellationToken ct) {\n"
     "        using (var s = Open()) {\n            await Flush(ct);\n        }\n        return await Compute(ct);\n    }\n"
     "    private static string Format(object o) => o.ToString();\n}\npublic interface IService {}\n",
     {"Service", "RunAsync", "Format", "IService"}, {"Open", "Flush", "Compute"}),
    ("main.kt", "suspend fun String.slugify(): String = lowercase()\ndata class Point(val x: Int)\n",
     {"slugify", "Point"}, set()),
    ("View.swift", "public func render(into ctx: Context) {}\nstruct Theme {}\n", {"render", "Theme"}, set()),
    ("ctl.php", "<?php\nclass Controller {\n    public static function index($req) {}\n}\n", {"Controller", "index"}, set()),
]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    server._fuzzy_cache.clear()
    server._symbol_cache.clear()
    yield
    server._fuzzy_cache.clear()
    server._symbol_cache.clear()


@pytest.mark.parametrize(("filename", "source", "wanted", "unwanted"), _FIXTURES, ids=[f[0] for f in _FIXTURES])
def test_definitions_indexed_statements_not(tmp_path: Path, filename, source, wanted, unwanted):
    (tmp_path / filename).write_text(source, encoding="utf-8")

    names = {name for name, _kind, rel in server._scan_symbols(str(tmp_path)) if rel == filename}

    assert wanted <= names, wanted - names
    assert not (unwanted & names), unwanted & names


def test_symbol_pick_is_a_file_ref_to_the_defining_file(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "repo.java").write_text("class A {\n    public void loadAccounts() {\n    }\n}\n")
    (tmp_path / "other.py").write_text("def unrelated(): ...\n")

    def items(word: str) -> list[dict]:
        resp = server.handle_request({"id": "1", "method": "complete.path", "params": {"word": word}})
        return resp["result"]["items"]

    hits = items("@symbol:loadacc")
    assert [(h["text"], h["display"], h["kind"]) for h in hits] == [
        ("@file:pkg/repo.java", "loadAccounts", "symbol")]
    assert "pkg/repo.java" in hits[0]["meta"]
    assert any(it["text"] == "@symbol:" for it in items("@"))
