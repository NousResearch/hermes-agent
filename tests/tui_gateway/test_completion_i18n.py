"""Stable presentation metadata without changing command or file ownership."""

import pytest

from tui_gateway import server


def test_complete_slash_tui_extra_exposes_stable_localization_key():
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/den"}}
    )
    density = next(item for item in resp["result"]["items"] if item["text"] == "/density")
    assert density["meta"] == "Toggle compact display mode"
    assert density["meta_key"] == "density"



def test_resolve_language_uses_shared_process_stable_contract(monkeypatch):
    from agent import i18n
    from hermes_cli import config as config_module

    current = {"display": {"language": "en"}}
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: current)
    i18n.reset_language_cache()

    try:
        assert server.resolve_language() == "en"
        current = {"display": {"language": "zh"}}
        assert server.resolve_language() == "en"

        i18n.reset_language_cache()
        assert server.resolve_language() == "zh"

        monkeypatch.setenv("HERMES_LANGUAGE", "pt_BR")
        monkeypatch.setattr(
            config_module,
            "load_config",
            lambda: pytest.fail("environment override must not read config"),
        )
        i18n.reset_language_cache()
        assert server.resolve_language() == "pt"
    finally:
        i18n.reset_language_cache()



def test_commands_catalog_exposes_stable_presentation_ids():
    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )
    tui = next(c for c in resp["result"]["categories"] if c["name"] == "TUI")
    density = next(item for item in tui["pairs"] if item[0] == "/density")
    assert density[1] == "Toggle compact display mode"
    assert tui["key"] == "tui"
    assert resp["result"]["description_keys"]["/density"] == "density"
    assert resp["result"]["description_keys"]["/new"] == "new"
    assert resp["result"]["description_keys"]["/bg"] == "bg"
    assert resp["result"]["description_keys"]["/btw"] == "btw"



def test_user_command_description_retains_its_ownership(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {
        "quick_commands": {"new": {"type": "exec", "command": "echo project", "description": "Project task"}}
    })
    result = server.handle_request({"id": "catalog", "method": "commands.catalog", "params": {}})["result"]
    assert dict(result["pairs"])["/new"] == "Project task"
    assert "/new" not in result["description_keys"]



def test_completion_metadata_does_not_classify_a_file_by_parent_name(tmp_path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "report.txt").write_text("example", encoding="utf-8")
    items = server.handle_request({
        "id": "path", "method": "complete.path",
        "params": {"word": "@file:report", "cwd": str(tmp_path)},
    })["result"]["items"]
    report = next(item for item in items if item["display"] == "report.txt")
    assert report["meta"] == "dir"
    assert "meta_key" not in report



def test_complete_slash_details_args():
    resp_root = server.handle_request(
        {"id": "0", "method": "complete.slash", "params": {"text": "/details"}}
    )
    resp_section = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/details t"}}
    )
    resp_mode = server.handle_request(
        {
            "id": "2",
            "method": "complete.slash",
            "params": {"text": "/details thinking e"},
        }
    )

    assert resp_root["result"]["replace_from"] == len("/details")
    assert any(item["text"] == " thinking" for item in resp_root["result"]["items"])
    assert any(item["text"] == "thinking" for item in resp_section["result"]["items"])
    assert any(item["text"] == "expanded" for item in resp_mode["result"]["items"])
    thinking = next(item for item in resp_root["result"]["items"] if item["text"] == " thinking")
    expanded = next(item for item in resp_mode["result"]["items"] if item["text"] == "expanded")
    assert thinking["meta_key"] == "completion.sectionOverride"
    assert expanded["meta_key"] == "completion.setSection"
    assert expanded["meta_vars"] == {"section": "thinking"}
