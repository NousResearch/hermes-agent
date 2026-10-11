"""Skill growth budgets must survive every write surface and profile switch (#130532)."""

import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.config import atomic_config_write


def _content(name, body="Keep lessons concise.\n"):
    return f"---\nname: {name}\ndescription: Use when testing skill growth.\n---\n# Rules\n{body}"


def _homes(tmp_path, monkeypatch, launch):
    root = tmp_path / ".hermes"
    homes = {name: root / "profiles" / name for name in ("a", "b")}
    launch_home = root if launch == "default" else homes["b"]
    launch_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
    return homes, launch_home


def _config(home, cap, warn=None, approval=False):
    atomic_config_write(home / "config.yaml", {
        "skills": {"max_skill_md_chars": cap, "size_warn_chars": warn,
                   "write_approval": approval, "ledger": False},
        "display": {"language": "en"},
    })


def _dispatch(payload):
    import tools.skill_manager_tool  # registers the production handler
    from tools.registry import registry
    return json.loads(registry.dispatch("skill_manage", payload))


def _write_op(mode, name, original, wanted):
    if mode == "create":
        return {"action": "create", "name": name, "content": wanted}
    if mode in ("rewrite", "edit"):
        return {"action": "edit" if mode == "edit" else "patch", "name": name, "content": wanted}
    if mode in ("write_root", "write_alias"):
        return {"action": "write_file", "name": name,
                "file_path": "./SKILL.md" if mode == "write_alias" else "SKILL.md",
                "file_content": wanted}
    op = {"action": "patch", "name": name, "old_string": original, "new_string": wanted}
    if mode == "patch_alias":
        op["file_path"] = "./SKILL.md"
    return op


def _apply(op, shape="flat", before_replay=None):
    payload = op if shape == "flat" else {"operations": [op]}
    result = _dispatch(payload)
    if shape == "replay":
        from tools.skill_manager_tool import apply_skill_pending
        from tools.write_approval import get_pending
        assert result["staged"] is True
        pending = get_pending("skills", result["pending_id"])
        if before_replay is not None:
            before_replay()
        result = json.loads(apply_skill_pending(pending["payload"]))
    return result


def _review_messages(payload, result):
    return [
        {"role": "assistant", "tool_calls": [{"id": "growth", "type": "function", "function": {
            "name": "skill_manage", "arguments": json.dumps(payload)}}]},
        {"role": "tool", "tool_call_id": "growth", "content": json.dumps(result)},
    ]


@pytest.mark.parametrize("launch", ["default", "named"])
@pytest.mark.parametrize("mode", ["create", "rewrite", "edit", "patch", "patch_alias",
                                 "write_root", "write_alias", "batch", "replay"])
def test_over_budget_growth_never_reaches_disk(tmp_path, monkeypatch, launch, mode):
    homes, launch_home = _homes(tmp_path, monkeypatch, launch)
    for index, profile in enumerate(("a", "b", "a")):
        home, name = homes[profile], f"growth-{index}"
        original = _content(name)
        wanted = original + "Learn another reusable rule.\n"
        cap = len(original) + 4
        _config(home, cap if profile == "a" else None, approval=mode == "replay")
        token = set_hermes_home_override(home)
        try:
            skill = home / "skills" / name
            if mode != "create":
                skill.mkdir(parents=True)
                (skill / "SKILL.md").write_text(original, encoding="utf-8")
            op = _write_op(mode, name, original, wanted)
            if mode == "batch":
                result = _dispatch({"operations": [
                    {"action": "write_file", "name": name, "file_path": "references/notes.md",
                     "file_content": "an independent supporting file"}, op,
                ]})
            elif mode == "replay":
                # A proposal staged without a cap must obey a budget lowered before approval.
                _config(home, None, approval=True)
                result = _apply(op, "replay", before_replay=lambda: _config(
                    home, cap if profile == "a" else None, approval=True))
            else:
                result = _apply(op)
            assert result["success"] is (profile == "b"), result
            if profile == "a":
                assert result["current_chars"] == (0 if mode == "create" else len(original))
                assert result["requested_delta"] == len(wanted) - result["current_chars"]
                assert result["cap"] == cap
                assert "references/" in result["error"]
                if mode == "create":
                    assert not skill.exists()
                else:
                    assert (skill / "SKILL.md").read_text() == original
                if mode == "batch":
                    assert not (skill / "references/notes.md").exists()
            else:
                assert (skill / "SKILL.md").read_text() == wanted
            # A supporting file named SKILL.md is outside the main-file budget.
            if mode != "create" or profile == "b":
                supporting = _apply({"action": "write_file", "name": name,
                    "file_path": "references/SKILL.md", "file_content": wanted * 2},
                    "replay" if mode == "replay" else "flat")
                assert supporting["success"] is True, supporting
                assert (skill / "references/SKILL.md").read_text() == wanted * 2
            # The exact boundary counts characters, not UTF-8 bytes.
            fit_name = f"fit-{index}"
            fitting = _content(fit_name, "é中😀\n")
            fitting += "x" * (cap - len(fitting))
            fitting_result = _apply({"action": "create", "name": fit_name, "content": fitting},
                                    "replay" if mode == "replay" else "batch")
            assert fitting_result["success"] is True, fitting_result
            assert len((home / "skills" / fit_name / "SKILL.md").read_text()) == cap
        finally:
            reset_hermes_home_override(token)
    if launch == "default":
        assert not (launch_home / "skills").exists()


@pytest.mark.parametrize("launch", ["default", "named"])
@pytest.mark.parametrize("shape", ["flat", "batch", "replay"])
@pytest.mark.parametrize("mode", ["patch", "rewrite", "write_root"])
def test_cleanup_and_size_warnings_follow_applied_writes(tmp_path, monkeypatch, launch, shape, mode):
    from agent.background_review import summarize_background_review_actions
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    import tools.skills_tool  # production read marks for the review fork
    from tools.registry import registry

    homes, _ = _homes(tmp_path, monkeypatch, launch)
    for index, profile in enumerate(("a", "b", "a")):
        home, name = homes[profile], f"cleanup-{index}"
        original = _content(name, "Keep lessons concise. Extra detail.\n")
        wanted = original.replace(" Extra detail.", " Detail.")
        threshold, cap = len(wanted) - 10, len(wanted) - 5
        _config(home, cap if profile == "a" else None,
                threshold if profile == "a" else None, approval=shape == "replay")
        token = set_hermes_home_override(home)
        origin = set_current_write_origin("background_review")
        try:
            skill = home / "skills" / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(original, encoding="utf-8")
            assert json.loads(registry.dispatch("skill_view", {"name": name}))["success"] is True
            op = _write_op(mode, name, original, wanted)
            payload = op if shape == "flat" else {"operations": [op]}
            result = _apply(op, shape)
            assert result["success"] is True, result  # still over the cap, but smaller
            assert (skill / "SKILL.md").read_text() == wanted
            row = result if shape == "flat" else result["results"][0]
            messages = _review_messages(payload, result)
            for notification in ("on", "verbose", "off"):
                lines = summarize_background_review_actions(messages, [], notification)
                expected = f"Skill '{name}' is at {len(wanted)} chars"
                assert any(expected in line for line in lines) is (profile == "a" and notification != "off")
            assert summarize_background_review_actions(messages, messages) == []
            if profile == "a":
                assert row["size_warning"] == {"name": name, "chars": len(wanted), "threshold": threshold}
            else:
                assert "size_warning" not in row
            # Equal-length maintenance also stays possible after lowering the budget.
            equal = wanted.replace("concise", "compact")
            assert _apply(_write_op(mode, name, wanted, equal), shape)["success"] is True
            grown = equal + "A new rule.\n"
            assert _apply(_write_op(mode, name, equal, grown), shape)["success"] is (profile == "b")
            assert summarize_background_review_actions(
                _review_messages(payload, {"success": False, "size_warning": row.get("size_warning")}), []) == []
            if shape != "flat":
                # A later shrink in one atomic batch must clear an intermediate size signal.
                current = (skill / "SKILL.md").read_text()
                intermediate = current.replace("Keep", "Use")
                final = _content(name, "")
                batch_payload = {"operations": [
                    _write_op("patch", name, current, intermediate),
                    _write_op("patch", name, intermediate, final),
                ]}
                batch_result = _dispatch(batch_payload)
                if shape == "replay":
                    from tools.skill_manager_tool import apply_skill_pending
                    from tools.write_approval import get_pending
                    pending = get_pending("skills", batch_result["pending_id"])
                    assert summarize_background_review_actions(_review_messages(batch_payload, batch_result), []) == []
                    batch_result = json.loads(apply_skill_pending(pending["payload"]))
                assert batch_result["success"] is True, batch_result
                lines = summarize_background_review_actions(_review_messages(batch_payload, batch_result), [])
                assert not any("is at" in line for line in lines)
            # Successful creates surface the warning even when the hard budget is disabled.
            fresh_name = f"fresh-{index}"
            fresh = _content(fresh_name, "")
            fresh += "x" * (cap - len(fresh))
            fresh_op = {"action": "create", "name": fresh_name, "content": fresh}
            fresh_result = _apply(fresh_op, shape)
            fresh_row = fresh_result if shape == "flat" else fresh_result["results"][0]
            assert fresh_result["success"] is True, fresh_result
            assert ("size_warning" in fresh_row) is (profile == "a")
        finally:
            reset_current_write_origin(origin)
            reset_hermes_home_override(token)
