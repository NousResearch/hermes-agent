"""End-to-end contracts for native memory governance plugin hooks."""

import json

from hermes_constants import get_hermes_home
from tools.memory_tool import MemoryStore


def test_real_plugin_governs_prompt_and_durable_write():
    """Discovery -> frozen context transform -> pre-write -> post-write."""
    import hermes_cli.plugins as plugins

    hermes_home = get_hermes_home()
    plugin_dir = hermes_home / "plugins" / "memory_governor"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: memory_governor\n"
        "provides_hooks:\n"
        "  - transform_memory_context\n"
        "  - pre_memory_write\n"
        "  - post_memory_write\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from hermes_constants import get_hermes_home\n"
        "\n"
        "def _context(target, entries, **kwargs):\n"
        "    if target != 'user':\n"
        "        return {'action': 'allow'}\n"
        "    return [entry for entry in entries if not entry.startswith('managed:')]\n"
        "\n"
        "def _pre(action, content=None, **kwargs):\n"
        "    if action == 'add' and content == 'draft fact':\n"
        "        return {'content': 'canonical fact'}\n"
        "    return {'action': 'allow'}\n"
        "\n"
        "def _post(action, target, entries, **kwargs):\n"
        "    marker = get_hermes_home() / 'memory-post-hook.txt'\n"
        "    marker.write_text(f'{action}:{target}:{entries[-1]}', encoding='utf-8')\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_hook('transform_memory_context', _context)\n"
        "    ctx.register_hook('pre_memory_write', _pre)\n"
        "    ctx.register_hook('post_memory_write', _post)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - memory_governor\n",
        encoding="utf-8",
    )
    memory_dir = hermes_home / "memories"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "USER.md").write_text(
        "managed: canonical elsewhere\n§\nnative preference",
        encoding="utf-8",
    )

    plugins._reset_plugin_managers_for_tests()
    store = MemoryStore()
    store.load_from_disk()

    prompt = store.format_for_system_prompt("user")
    assert "native preference" in prompt
    assert "managed: canonical elsewhere" not in prompt
    assert store.user_entries == ["managed: canonical elsewhere", "native preference"]

    result = store.add("memory", "draft fact")

    assert result["success"] is True
    assert store.memory_entries == ["canonical fact"]
    assert (hermes_home / "memory-post-hook.txt").read_text(encoding="utf-8") == (
        "add:memory:canonical fact"
    )
    plugins._reset_plugin_managers_for_tests()


def test_real_governor_failure_blocks_write_and_raw_prompt_injection():
    """Discovery -> callback failure -> both governance boundaries fail closed."""
    import hermes_cli.plugins as plugins

    hermes_home = get_hermes_home()
    plugin_dir = hermes_home / "plugins" / "failing_memory_governor"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: failing_memory_governor\n"
        "provides_hooks:\n"
        "  - transform_memory_context\n"
        "  - pre_memory_write\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def fail(*args, **kwargs):\n"
        "    raise RuntimeError('governance backend unavailable')\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_hook('transform_memory_context', fail)\n"
        "    ctx.register_hook('pre_memory_write', fail)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - failing_memory_governor\n",
        encoding="utf-8",
    )
    memory_dir = hermes_home / "memories"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "USER.md").write_text("private native entry", encoding="utf-8")

    plugins._reset_plugin_managers_for_tests()
    store = MemoryStore()
    store.load_from_disk()

    assert store.format_for_system_prompt("user") is None
    result = store.add("memory", "must remain blocked")
    assert result == {
        "success": False,
        "error": "Memory write blocked because a governance plugin failed.",
    }
    assert store.memory_entries == []
    assert not (memory_dir / "MEMORY.md").exists()
    plugins._reset_plugin_managers_for_tests()


def test_real_governor_abstention_blocks_write_and_raw_prompt_injection():
    """A registered governor must explicitly transform, allow, skip, or block."""
    import hermes_cli.plugins as plugins

    hermes_home = get_hermes_home()
    plugin_dir = hermes_home / "plugins" / "abstaining_memory_governor"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: abstaining_memory_governor\n"
        "provides_hooks:\n"
        "  - transform_memory_context\n"
        "  - pre_memory_write\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def abstain(*args, **kwargs):\n"
        "    return None\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_hook('transform_memory_context', abstain)\n"
        "    ctx.register_hook('pre_memory_write', abstain)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - abstaining_memory_governor\n",
        encoding="utf-8",
    )
    memory_dir = hermes_home / "memories"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "USER.md").write_text("private native entry", encoding="utf-8")

    plugins._reset_plugin_managers_for_tests()
    store = MemoryStore()
    store.load_from_disk()

    assert store.format_for_system_prompt("user") is None
    result = store.add("memory", "must require an explicit decision")
    assert result == {
        "success": False,
        "error": "Memory write blocked because a governance plugin failed.",
    }
    assert store.memory_entries == []
    assert not (memory_dir / "MEMORY.md").exists()
    plugins._reset_plugin_managers_for_tests()


def test_real_plugin_routes_shared_home_writes_by_window_provenance():
    """One shared native home can route different UI windows externally."""
    import hermes_cli.plugins as plugins

    hermes_home = get_hermes_home()
    plugin_dir = hermes_home / "plugins" / "window_memory_router"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: window_memory_router\n"
        "provides_hooks:\n"
        "  - pre_memory_write\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "import json\n"
        "from hermes_constants import get_hermes_home\n"
        "\n"
        "def route(action, target, content=None, provenance=None, **kwargs):\n"
        "    provenance = provenance or {}\n"
        "    route_id = provenance.get('ui_session_id') or provenance.get('session_id')\n"
        "    operation_id = provenance.get('operation_id')\n"
        "    if not route_id or not operation_id:\n"
        "        return {'action': 'block', 'message': 'write provenance required'}\n"
        "    path = get_hermes_home() / f'routed-{route_id}.json'\n"
        "    path.write_text(json.dumps({\n"
        "        'operation_id': operation_id,\n"
        "        'action': action,\n"
        "        'target': target,\n"
        "        'content': content,\n"
        "    }), encoding='utf-8')\n"
        "    return {'action': 'skip', 'message': 'routed externally'}\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_hook('pre_memory_write', route)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - window_memory_router\n",
        encoding="utf-8",
    )

    plugins._reset_plugin_managers_for_tests()
    store = MemoryStore()
    store.load_from_disk()

    first = store.add(
        "memory",
        "desktop fact",
        provenance={
            "session_id": "shared-conversation",
            "ui_session_id": "desktop-window",
            "operation_id": "write-desktop-1",
        },
    )
    second = store.add(
        "memory",
        "qq fact",
        provenance={
            "session_id": "shared-conversation",
            "ui_session_id": "qq-window",
            "operation_id": "write-qq-1",
        },
    )

    assert first["native_write"] is False
    assert second["native_write"] is False
    assert store.memory_entries == []
    assert json.loads(
        (hermes_home / "routed-desktop-window.json").read_text(encoding="utf-8")
    ) == {
        "operation_id": "write-desktop-1",
        "action": "add",
        "target": "memory",
        "content": "desktop fact",
    }
    assert json.loads(
        (hermes_home / "routed-qq-window.json").read_text(encoding="utf-8")
    ) == {
        "operation_id": "write-qq-1",
        "action": "add",
        "target": "memory",
        "content": "qq fact",
    }
    plugins._reset_plugin_managers_for_tests()


def test_approved_replay_still_crosses_governance_boundary(monkeypatch, tmp_path):
    """The approval replay helper must not bypass the store-level hook."""
    from tools.memory_tool import apply_memory_pending

    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook", lambda name: name == "pre_memory_write"
    )
    calls = []

    def govern(name, **kwargs):
        calls.append((name, kwargs))
        return [{"content": "canonical approved fact"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", govern)
    store = MemoryStore()
    store.load_from_disk()

    result = apply_memory_pending(
        {
            "action": "add",
            "target": "memory",
            "content": "staged fact",
            "provenance": {
                "operation_id": "approved-write-1",
                "session_id": "origin-session",
                "ui_session_id": "origin-window",
            },
        },
        store,
    )

    assert result["success"] is True
    assert store.memory_entries == ["canonical approved fact"]
    assert calls[0][0] == "pre_memory_write"
    assert calls[0][1]["provenance"] == {
        "operation_id": "approved-write-1",
        "session_id": "origin-session",
        "ui_session_id": "origin-window",
    }
