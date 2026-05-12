import json
import tools.translation_master_tool as tmt


def test_tm_translate_requires_texts():
    out = json.loads(tmt.tm_translate({"texts": []}))
    assert "error" in out


def test_tm_translate_missing_node(monkeypatch):
    monkeypatch.setattr(tmt, "_node_bin", lambda: None)
    out = json.loads(tmt.tm_translate({"texts": ["Hallo"], "target_locale": "en"}))
    assert "error" in out


def test_tm_translate_missing_pkg(monkeypatch):
    monkeypatch.setattr(tmt, "_node_bin", lambda: "/usr/bin/node")

    def fake_run(script, env=None):
        return {"error": "Cannot find module '@translation-master/chrome'"}

    monkeypatch.setattr(tmt, "_run_node", fake_run)
    out = json.loads(tmt.tm_translate({"texts": ["Hallo"], "target_locale": "en"}))
    assert "error" in out
    assert "@translation-master/chrome not installed" in out["error"]
