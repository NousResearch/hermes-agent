"""Regression for #123190: path placeholders are references, shell writes are not."""

import pytest

from tools.plugin_guard import scan_plugin, should_allow_plugin_install
from tools.skills_guard import scan_skill


CONFIG_TARGETS = [
    (".claude/settings.json", "other_agent_config"),
    (".codex/config.toml", "other_agent_config"),
    (".hermes/config.yaml", "hermes_config"),
    ("AGENTS.md", "agent_config"),
]


@pytest.mark.parametrize("target,family", CONFIG_TARGETS)
@pytest.mark.parametrize("placeholder", [
    "vault", "vault-root", "vault_root", "vault.path", "path/to/vault", "vault:path",
])
def test_path_placeholders_allow_install_and_keep_reference(tmp_path, target, family, placeholder):
    from hermes_cli.plugins_cmd import _scan_plugin_tree

    content = f"Settings live in <{placeholder}>/{target} and are read only.\n"
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: demo\nversion: 0.0.1\n", encoding="utf-8")
    (plugin / "README.md").write_text(content, encoding="utf-8")
    (plugin / "reference.ts").write_text("// " + content, encoding="utf-8")
    result = _scan_plugin_tree(plugin, "owner/demo", force=False)
    assert result.verdict == "safe"
    assert should_allow_plugin_install(result)[0] is True
    assert {f.file for f in result.findings if f.pattern_id == family + "_ref"} == {
        "README.md", "reference.ts",
    }
    assert not any(f.pattern_id == family + "_mod_shell" for f in result.findings)

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(content, encoding="utf-8")
    result = scan_skill(skill)
    assert result.verdict == "safe"
    assert any(f.pattern_id == family + "_ref" for f in result.findings)
    assert not any(f.pattern_id == family + "_mod_shell" for f in result.findings)


@pytest.mark.parametrize("target,family", CONFIG_TARGETS)
@pytest.mark.parametrize("command", [
    "echo payload > {target}",
    "echo payload>{target}",
    "echo payload >> {target}",
    'printf "payload">{target}',
    "printf 'payload'>{target}",
    "(printf payload)>{target}",
    "echo payload > <vault>/{target}",
    "echo payload><vault-root>/{target}",
    "echo payload<input>{target}",
    "cat <input>{target}",
    "cat <input>~/{target}",
    'echo payload > "{target}"',
    "Settings live in <vault>/{target}; echo payload>{target}",
    "echo payload>{target}; settings live in <vault>/{target}",
    "echo payload | tee {target}",
    "sed -i 's/old/new/' {target}",
    "cp payload {target}",
    "mv payload {target}",
])
def test_real_config_writes_remain_critical_and_block_install(tmp_path, target, family, command):
    from hermes_cli.plugins_cmd import PluginScanBlocked, _scan_plugin_tree

    content = command.format(target=target) + "\n"
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: demo\nversion: 0.0.1\n", encoding="utf-8")
    (plugin / "README.md").write_text(content, encoding="utf-8")
    result = scan_plugin(plugin)
    assert any(f.pattern_id == family + "_mod_shell" and f.severity == "critical"
               for f in result.findings)
    assert result.verdict == "dangerous"
    assert should_allow_plugin_install(result, force=True)[0] is False
    with pytest.raises(PluginScanBlocked):
        _scan_plugin_tree(plugin, "owner/demo", force=True)

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(content, encoding="utf-8")
    result = scan_skill(skill)
    assert any(f.pattern_id == family + "_mod_shell" and f.severity == "critical"
               for f in result.findings)
    assert result.verdict == "dangerous"
