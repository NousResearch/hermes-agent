import ast
import asyncio
import threading
import time
import textwrap
from pathlib import Path

import hermes_state

from hermes_cli import web_server


SESSIONDB_READ_METHODS = {
    "get_session",
    "get_session_title",
    "list_sessions_rich",
    "message_count",
    "search_messages",
    "search_sessions_by_id",
    "session_count",
    "session_counts_by_source",
}

OFFLOAD_CALL_NAMES = {
    "_blocking_io",
    "_session_db_read",
    "run_in_executor",
    "to_thread",
}

# Pass-4 affinity enumeration record for the PRD's nine SessionDB read sites,
# plus the paired ``get_sessions`` total count that shares that offloaded page
# helper. Each site opens a request-local SessionDB inside a synchronous helper
# that the async route awaits via the executor. No site reuses a loop-thread
# connection.
PRD_NINE_SITE_AFFINITY_RECORD = {
    "get_status:list_sessions_rich(limit=50)": "_read_active_session_count -> _open_session_db_for_profile",
    "get_sessions:list_sessions_rich(page)": "_read_sessions_page -> _open_session_db_for_profile",
    "get_sessions:session_count(total)": "_read_sessions_page -> _open_session_db_for_profile",
    "search_sessions:search_sessions_by_id": "_read_session_search -> _open_session_db_for_profile",
    "search_sessions:search_sessions_by_title": "_read_session_search -> _open_session_db_for_profile",
    "search_sessions:get_session(compression_root/current)": "_read_session_search -> _open_session_db_for_profile",
    "search_sessions:get_session(compression_root/parent)": "_read_session_search -> _open_session_db_for_profile",
    "search_sessions:search_messages(FTS)": "_read_session_search -> _open_session_db_for_profile",
    "get_session_detail:get_session": "_read_session_detail -> _open_session_db_for_profile",
    "rename_session_endpoint:get_session_title": "_rename_session_record -> _open_session_db_for_profile",
    "get_session_stats:session_counts_by_source": "_read_session_stats -> _open_session_db_for_profile",
}


def test_affinity_record_covers_prd_sites_and_paired_total_count():
    assert len(PRD_NINE_SITE_AFFINITY_RECORD) == 11
    assert all(
        target.endswith("_open_session_db_for_profile")
        for target in PRD_NINE_SITE_AFFINITY_RECORD.values()
    )


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _func_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _nearest_callable(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.AsyncFunctionDef | ast.FunctionDef | ast.Lambda | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda)):
            return current
    return None


def _enclosing_async(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.AsyncFunctionDef | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.AsyncFunctionDef):
            return current
    return None


def _is_offload_call(call: ast.Call) -> bool:
    return _func_name(call.func) in OFFLOAD_CALL_NAMES


def _lambda_is_offloaded(lambda_node: ast.Lambda, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(lambda_node)
    return isinstance(parent, ast.Call) and _is_offload_call(parent)


def _named_inner_is_offloaded(inner: ast.FunctionDef, async_fn: ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(async_fn):
        if isinstance(node, ast.Call) and _is_offload_call(node):
            if any(isinstance(arg, ast.Name) and arg.id == inner.name for arg in node.args):
                return True
    return False


def _bare_sessiondb_reads(source: str) -> list[tuple[int, str, str]]:
    tree = ast.parse(textwrap.dedent(source))
    parents = _parent_map(tree)
    violations: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in SESSIONDB_READ_METHODS:
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "db":
            continue

        owner = _nearest_callable(node, parents)
        if owner is None:
            continue
        if isinstance(owner, ast.AsyncFunctionDef):
            violations.append((node.lineno, owner.name, node.func.attr))
            continue

        async_fn = _enclosing_async(owner, parents)
        if async_fn is None:
            # Top-level sync helpers are safe when the async route awaits a
            # single executor offload around the helper call.
            continue
        if isinstance(owner, ast.Lambda) and _lambda_is_offloaded(owner, parents):
            continue
        if isinstance(owner, ast.FunctionDef) and _named_inner_is_offloaded(owner, async_fn):
            continue
        violations.append((node.lineno, async_fn.name, node.func.attr))

    return violations


def test_ast_lint_self_test_distinguishes_bare_reads_from_offloaded_forms():
    source = """
        import asyncio
        import functools

        async def bad(db):
            return db.get_session("x")

        async def ok_lambda(db, loop):
            return await loop.run_in_executor(None, lambda: db.get_session("x"))

        async def ok_partial(db, loop):
            return await loop.run_in_executor(None, functools.partial(db.get_session, "x"))

        async def ok_inner_def(db, loop):
            def _read():
                return db.get_session("x")
            return await loop.run_in_executor(None, _read)

        async def ok_to_thread(db):
            return await asyncio.to_thread(db.get_session, "x")

        def sync_helper(db):
            return db.get_session("x")
    """

    assert _bare_sessiondb_reads(source) == [(6, "bad", "get_session")]


def test_web_server_has_no_bare_sessiondb_reads_in_async_handlers():
    source = Path(web_server.__file__).read_text(encoding="utf-8")
    assert _bare_sessiondb_reads(source) == []


def test_session_db_heavy_reads_are_limited_to_two_executor_calls(monkeypatch):
    in_flight = 0
    max_in_flight = 0

    async def fake_blocking_io(fn, *args):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.01)
            return fn(*args)
        finally:
            in_flight -= 1

    monkeypatch.setattr(web_server, "_blocking_io", fake_blocking_io)

    async def run_reads():
        return await asyncio.gather(
            *(
                web_server._session_db_read(lambda value=value: value, heavy=True)
                for value in range(3)
            )
        )

    assert asyncio.run(run_reads()) == [0, 1, 2]
    assert max_in_flight == 2


def test_get_status_offloads_active_session_scan(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _GatewayConfig:
        def get_connected_platforms(self):
            return []

    class _DB:
        def list_sessions_rich(self, *, limit):
            assert limit == 50
            db_threads.append(threading.get_ident())
            return [{"ended_at": None, "last_active": time.time()}]

        def close(self):
            db_threads.append(threading.get_ident())

    import gateway.config as gateway_config

    monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
    monkeypatch.setattr(web_server, "get_running_pid", lambda: None)
    monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)
    monkeypatch.setattr(web_server, "read_runtime_status", lambda: None)
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: _GatewayConfig())
    monkeypatch.setattr(web_server, "_resolve_restart_drain_timeout", lambda: 0)
    monkeypatch.setattr(web_server, "_dashboard_local_update_managed_externally", lambda: False)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(hermes_state, "SessionDB", _DB)

    status = asyncio.run(web_server.get_status())

    assert status["active_sessions"] == 1
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)


def test_get_sessions_offloads_sessiondb_page(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _DB:
        def _record(self):
            db_threads.append(threading.get_ident())

        def list_sessions_rich(self, **kwargs):
            self._record()
            assert kwargs["limit"] == 5
            assert kwargs["offset"] == 1
            assert kwargs["order_by_last_active"] is True
            return [{"id": "s1", "ended_at": None, "last_active": 0, "archived": 0, "pinned": 0}]

        def session_count(self, **kwargs):
            self._record()
            assert kwargs["exclude_children"] is True
            return 1

        def close(self):
            self._record()

    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile=None: _DB())

    result = asyncio.run(web_server.get_sessions(limit=5, offset=1, order="recent"))

    assert result == {
        "sessions": [
            {
                "id": "s1",
                "ended_at": None,
                "last_active": 0,
                "archived": False,
                "pinned": False,
                "is_active": False,
            }
        ],
        "total": 1,
        "limit": 5,
        "offset": 1,
    }
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)


def test_search_sessions_offloads_all_sessiondb_reads(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _DB:
        def _record(self):
            db_threads.append(threading.get_ident())

        def search_sessions_by_id(self, query, *, limit, include_archived):
            self._record()
            assert query == "root"
            assert limit == 2
            assert include_archived is True
            return [
                {
                    "id": "root",
                    "preview": "",
                    "source": "cli",
                    "model": "m",
                    "started_at": 1,
                }
            ]

        def search_sessions_by_title(self, query, *, limit, include_archived):
            self._record()
            assert query == "root"
            assert limit == 2
            assert include_archived is True
            return []

        def search_messages(self, *, query, limit):
            self._record()
            assert query == "root*"
            assert limit == 50
            return [
                {
                    "session_id": "child",
                    "snippet": "hit",
                    "role": "user",
                    "source": "cli",
                    "model": "m",
                    "session_started": 2,
                }
            ]

        def get_session(self, session_id):
            self._record()
            return {"id": session_id, "parent_session_id": None, "started_at": 1}

        def get_compression_tip(self, session_id):
            self._record()
            return session_id

        def close(self):
            self._record()

    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile=None: _DB())

    result = asyncio.run(web_server.search_sessions(q="root", limit=2))

    assert result == {
        "results": [
            {
                "snippet": "Session ID: root",
                "role": None,
                "source": "cli",
                "model": "m",
                "session_started": 1,
                "session_id": "root",
                "lineage_root": "root",
            },
            {
                "snippet": "hit",
                "role": "user",
                "source": "cli",
                "model": "m",
                "session_started": 2,
                "session_id": "child",
                "lineage_root": "child",
            },
        ]
    }
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)


def test_get_session_detail_offloads_sessiondb_lookup(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _DB:
        def _record(self):
            db_threads.append(threading.get_ident())

        def resolve_session_id(self, session_id):
            self._record()
            assert session_id == "s"
            return "sid"

        def get_session(self, session_id):
            self._record()
            assert session_id == "sid"
            return {"id": "sid", "title": "Session"}

        def close(self):
            self._record()

    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile=None: _DB())

    assert asyncio.run(web_server.get_session_detail("s")) == {"id": "sid", "title": "Session"}
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)


def test_rename_session_offloads_sessiondb_update_and_title_read(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _DB:
        def _record(self):
            db_threads.append(threading.get_ident())

        def resolve_session_id(self, session_id):
            self._record()
            assert session_id == "s"
            return "sid"

        def set_session_title(self, session_id, title):
            self._record()
            assert (session_id, title) == ("sid", "New title")

        def set_session_archived(self, session_id, archived):
            self._record()
            assert (session_id, archived) == ("sid", True)

        def set_session_pinned(self, session_id, pinned):
            self._record()
            assert (session_id, pinned) == ("sid", True)

        def get_session_title(self, session_id):
            self._record()
            assert session_id == "sid"
            return "New title"

        def close(self):
            self._record()

    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile=None: _DB())

    result = asyncio.run(
        web_server.rename_session_endpoint(
            "s",
            web_server.SessionRename(title="New title", archived=True, pinned=True),
        )
    )

    assert result == {"ok": True, "title": "New title", "archived": True, "pinned": True}
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)
