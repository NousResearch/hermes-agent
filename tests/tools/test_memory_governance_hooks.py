"""End-to-end contracts for native memory governance plugin hooks."""

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
        "        return None\n"
        "    return [entry for entry in entries if not entry.startswith('managed:')]\n"
        "\n"
        "def _pre(action, content=None, **kwargs):\n"
        "    if action == 'add' and content == 'draft fact':\n"
        "        return {'content': 'canonical fact'}\n"
        "    return None\n"
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


def test_approved_replay_still_crosses_governance_boundary(monkeypatch, tmp_path):
    """The approval replay helper must not bypass the store-level hook."""
    from tools.memory_tool import apply_memory_pending

    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook", lambda name: name == "pre_memory_write"
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: [{"content": "canonical approved fact"}],
    )
    store = MemoryStore()
    store.load_from_disk()

    result = apply_memory_pending(
        {"action": "add", "target": "memory", "content": "staged fact"},
        store,
    )

    assert result["success"] is True
    assert store.memory_entries == ["canonical approved fact"]
