"""session_search on a gateway shared by many people: a plugin's judgement decides which past
sessions this turn's reader may see, in every shape, and anything it cannot decide stays hidden."""
import contextvars
import dataclasses
import json
import threading
import time

import pytest
import yaml

from agent.session_visibility import SessionJudgement, SessionPlace, SessionVisibility, ToolCall
from hermes_state import SessionDB
from tools import session_search_tool
from tools.session_search_tool import session_search

READER = contextvars.ContextVar("reader", default=None)


class Rule(SessionJudgement):
    def __init__(self, visible=(), shown_tools=None, fail_on=(), hang_on=(), answer=True):
        self.visible, self.shown_tools = set(visible), shown_tools
        self.fail_on, self.hang_on, self.answer = set(fail_on), set(hang_on), answer
        self.places, self.calls, self.readers, self.finished = [], [], [], 0
        self.release = threading.Event()

    def may_see(self, place):
        self.places.append(place)
        self.readers.append(READER.get())
        if place.id in self.fail_on:
            raise RuntimeError("membership lookup failed")
        if place.id in self.hang_on:
            self.release.wait(5)
        return self.answer if place.id in self.visible else False

    def finish(self):
        self.finished += 1


class RedactingRule(Rule):
    def may_show_tool_result(self, place, call):
        self.calls.append((place.id, call))
        if call.name == "explode":
            raise RuntimeError("platform lookup failed")
        return call.name in (self.shown_tools or ())


class Provider(SessionVisibility):
    name = "test-visibility"

    def __init__(self, judgement=None, fails=False):
        self._judgement, self._fails = judgement, fails

    def judgement(self):
        if self._fails:
            raise RuntimeError("speaker unavailable")
        return self._judgement


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def visibility(monkeypatch):
    def install(provider=None, require=False, timeout=2.0):
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_session_visibility", lambda: provider)
        monkeypatch.setattr("agent.session_visibility.visibility_settings", lambda: {
            "require_visibility": require, "visibility_timeout_seconds": timeout})
        return provider
    return install


def _session(db, sid, *, started, title=None, parent=None, chat_id="C1", chat_type="group", user_id="U1",
             source="slack", end_reason=None, lines=()):
    db.create_session(sid, source=source, user_id=user_id, chat_id=chat_id, chat_type=chat_type,
                      thread_id="1700000000.000100", parent_session_id=parent)
    db._conn.execute("UPDATE sessions SET started_at = ?, title = ?, end_reason = ? WHERE id = ?",
                     (started, title, end_reason, sid))
    return [db.append_message(sid, role=role, content=text) for role, text in lines]


def _seed_budget(db):
    now = int(time.time())
    _session(db, "s_open", started=now - 300, title="Budget review",
             lines=[("user", "What is the marketing budget?"), ("assistant", "The marketing budget is 40k.")])
    _session(db, "s_private", started=now - 200, chat_id="C2",
             lines=[("user", "Confidential marketing budget cut"), ("assistant", "Marketing budget cut to 10k.")])
    _session(db, "s_dm", started=now - 100, chat_id="D1", chat_type="dm",
             lines=[("user", "My marketing budget question"), ("assistant", "Your marketing budget answer.")])


def _sessions_in(payload):
    return [r["session_id"] for r in json.loads(payload).get("results", [])]


def _same_answer(hidden, hidden_id, missing, missing_id):
    return hidden.replace(hidden_id, "<id>") == missing.replace(missing_id, "<id>")


def test_without_a_provider_or_requirement_every_session_is_searchable(db, visibility):
    _seed_budget(db)
    visibility()
    assert set(_sessions_in(session_search(query="marketing budget", limit=5, db=db))) == {"s_open", "s_private", "s_dm"}


def test_redteam_required_without_a_provider_hides_every_session(db, visibility):
    _seed_budget(db)
    visibility(require=True)
    assert _sessions_in(session_search(db=db)) == []
    assert _sessions_in(session_search(query="marketing budget", limit=5, db=db)) == []
    assert _same_answer(session_search(session_id="s_open", db=db), "s_open",
                        session_search(session_id="s_nowhere", db=db), "s_nowhere")


def test_discovery_returns_only_visible_sessions_and_hidden_ones_take_no_slot(db, visibility):
    _seed_budget(db)
    visibility(Provider(Rule(visible={"s_open", "s_dm"})))
    assert set(_sessions_in(session_search(query="marketing budget", limit=2, db=db))) == {"s_open", "s_dm"}


def test_redteam_discovery_with_only_hidden_matches_answers_like_no_match(db, visibility):
    _seed_budget(db)
    visibility(Provider(Rule(visible=set())))
    hidden = json.loads(session_search(query="confidential", db=db))
    visibility()
    missing = json.loads(session_search(query="nonexistentword", db=db))
    assert hidden["results"] == missing["results"] == [] and hidden.get("message") == missing.get("message")


def test_redteam_a_hit_in_a_child_needs_its_lineage_root_visible_too(db, visibility):
    now = int(time.time())
    _session(db, "s_root", started=now - 500, end_reason="compression", lines=[("user", "kickoff")])
    _session(db, "s_child", started=now - 400, parent="s_root",
             lines=[("user", "the quarterly forecast"), ("assistant", "Quarterly forecast is up.")])
    rule = visibility(Provider(Rule(visible={"s_child"})))._judgement
    assert _sessions_in(session_search(query="quarterly forecast", db=db)) == []
    assert {p.id for p in rule.places} >= {"s_child", "s_root"}


def test_redteam_a_hidden_title_match_does_not_surface(db, visibility):
    _seed_budget(db)
    visibility(Provider(Rule(visible={"s_dm"})))
    payload = json.loads(session_search(query="Budget review", db=db))
    assert all(r["session_id"] != "s_open" and r["matched_role"] != "session_title" for r in payload["results"])


def test_browse_lists_only_visible_sessions(db, visibility):
    _seed_budget(db)
    visibility(Provider(Rule(visible={"s_open"})))
    assert _sessions_in(session_search(db=db)) == ["s_open"]


def test_redteam_reading_a_hidden_session_answers_like_a_missing_one(db, visibility):
    _seed_budget(db)
    visibility(Provider(Rule(visible={"s_open"})))
    hidden, missing = session_search(session_id="s_private", db=db), session_search(session_id="s_gone", db=db)
    assert _same_answer(hidden, "s_private", missing, "s_gone")
    assert "10k" not in hidden
    assert json.loads(session_search(session_id="s_open", db=db))["mode"] == "read"


def test_redteam_scrolling_a_hidden_session_answers_like_a_missing_one(db, visibility):
    _seed_budget(db)
    anchor = _sessions_anchor(db, "s_private")
    visibility(Provider(Rule(visible={"s_open"})))
    hidden = session_search(session_id="s_private", around_message_id=anchor, db=db)
    missing = session_search(session_id="s_gone", around_message_id=anchor, db=db)
    assert _same_answer(hidden, "s_private", missing, "s_gone") and "10k" not in hidden


def _sessions_anchor(db, sid):
    return db.get_messages(sid)[0]["id"]


def test_redteam_scroll_does_not_rebind_into_a_hidden_child(db, visibility):
    now = int(time.time())
    _session(db, "s_parent", started=now - 500, end_reason="compression", lines=[("user", "kickoff")])
    child_ids = _session(db, "s_kid", started=now - 400, parent="s_parent",
                         lines=[("user", "salary bands"), ("assistant", "Band C tops at 180k.")])
    visibility(Provider(Rule(visible={"s_parent"})))
    answer = session_search(session_id="s_parent", around_message_id=child_ids[1], db=db)
    assert "180k" not in answer and json.loads(answer)["success"] is False


def test_redteam_another_profiles_sessions_are_judged_too(db, tmp_path, visibility, monkeypatch):
    other = SessionDB(tmp_path / "other.db")
    _session(other, "s_elsewhere", started=int(time.time()), lines=[("user", "launch codes"), ("assistant", "1234")])
    monkeypatch.setattr(session_search_tool, "_resolve_profile_db", lambda profile: other if profile else None)
    rule = visibility(Provider(Rule(visible=set())))._judgement
    assert "1234" not in session_search(session_id="s_elsewhere", profile="ops", db=db)
    assert _sessions_in(session_search(profile="ops", db=db)) == []
    assert "s_elsewhere" in {p.id for p in rule.places}


def test_redteam_a_judgement_that_raises_hides_that_session(db, visibility):
    _seed_budget(db)
    visibility(Provider(Rule(visible={"s_open", "s_private"}, fail_on={"s_private"})))
    assert set(_sessions_in(session_search(query="marketing budget", limit=5, db=db))) == {"s_open"}


@pytest.mark.parametrize("answer", [1, "yes", None, object()])
def test_redteam_only_a_literal_true_shows_a_session(db, visibility, answer):
    _seed_budget(db)
    visibility(Provider(Rule(visible={"s_open"}, answer=answer)))
    assert _sessions_in(session_search(db=db)) == []


def test_redteam_a_judgement_that_hangs_hides_the_rest_of_the_call(db, visibility):
    _seed_budget(db)
    rule = Rule(visible={"s_open", "s_private", "s_dm"}, hang_on={"s_dm"})
    visibility(Provider(rule), timeout=0.2)
    try:
        started = time.monotonic()
        assert _sessions_in(session_search(db=db)) == []
        assert time.monotonic() - started < 2
        assert rule.finished == 1
    finally:
        rule.release.set()


def test_redteam_a_failing_judgement_factory_hides_everything(db, visibility):
    _seed_budget(db)
    visibility(Provider(fails=True))
    assert _sessions_in(session_search(db=db)) == []
    assert _sessions_in(session_search(query="marketing budget", db=db)) == []


def test_the_judgement_learns_only_where_a_session_happened(db, visibility):
    _seed_budget(db)
    rule = visibility(Provider(Rule(visible={"s_open"})))._judgement
    session_search(session_id="s_open", db=db)
    assert [f.name for f in dataclasses.fields(SessionPlace)] == [
        "id", "source", "user_id", "chat_id", "chat_type", "thread_id", "parent_session_id"]
    assert rule.places[0] == SessionPlace(id="s_open", source="slack", user_id="U1", chat_id="C1",
                                          chat_type="group", thread_id="1700000000.000100",
                                          parent_session_id=None)


def test_the_judgement_runs_in_the_callers_context(db, visibility):
    _seed_budget(db)
    rule = visibility(Provider(Rule(visible={"s_open"})))._judgement
    token = READER.set("U_reader")
    try:
        session_search(db=db)
    finally:
        READER.reset(token)
    assert rule.readers and set(rule.readers) == {"U_reader"}


def test_finish_runs_once_per_call_even_when_the_judgement_fails(db, visibility):
    _seed_budget(db)
    rule = visibility(Provider(Rule(visible={"s_open"}, fail_on={"s_dm"})))._judgement
    session_search(query="marketing budget", limit=5, db=db)
    session_search(session_id="s_dm", db=db)
    assert rule.finished == 2


def _seed_tool_session(db):
    now = int(time.time())
    _session(db, "s_tools", started=now - 50)
    db.append_message("s_tools", role="user", content="pull the payroll sheet")
    db.append_message("s_tools", role="assistant", content="", tool_calls=[
        {"id": "call_pay", "type": "function", "function": {"name": "hub_run", "arguments": '{"operation": "sheets.read"}'}},
        {"id": "call_web", "type": "function", "function": {"name": "web_search", "arguments": '{"q": "payroll"}'}}])
    db.append_message("s_tools", role="tool", content="payroll: alice 9000, bob 8000", tool_name="hub_run",
                      tool_call_id="call_pay")
    db.append_message("s_tools", role="tool", content="payroll laws overview", tool_name="web_search",
                      tool_call_id="call_web")
    db.append_message("s_tools", role="assistant", content="The payroll sheet is loaded.")


def test_tool_results_of_visible_sessions_are_shown_by_default(db, visibility):
    _seed_tool_session(db)
    visibility(Provider(Rule(visible={"s_tools"})))
    assert "alice 9000" in session_search(session_id="s_tools", db=db)


def test_a_withheld_tool_result_is_replaced_in_read_scroll_and_discovery(db, visibility):
    _seed_tool_session(db)
    rule = visibility(Provider(RedactingRule(visible={"s_tools"}, shown_tools={"web_search"})))._judgement
    anchor = _sessions_anchor(db, "s_tools")
    for payload in (session_search(session_id="s_tools", db=db),
                    session_search(session_id="s_tools", around_message_id=anchor, window=10, db=db)):
        assert "alice 9000" not in payload and "payroll laws overview" in payload
    assert "alice 9000" not in session_search(query="alice", role_filter="tool", detail="full", db=db)
    assert "payroll laws overview" in session_search(query="overview", role_filter="tool", detail="full", db=db)
    assert ("s_tools", ToolCall(name="hub_run", arguments='{"operation": "sheets.read"}')) in rule.calls


def test_redteam_a_withheld_tool_hit_loses_its_snippet(db, visibility):
    _seed_tool_session(db)
    visibility(Provider(RedactingRule(visible={"s_tools"}, shown_tools=set())))
    payload = session_search(query="alice", role_filter="tool", detail="full", db=db)
    assert "alice" not in payload.replace('"query": "alice"', "")


def test_redteam_a_tool_result_judgement_that_raises_withholds_it(db, visibility):
    now = int(time.time())
    _session(db, "s_boom", started=now)
    db.append_message("s_boom", role="assistant", content="", tool_calls=[
        {"id": "call_x", "type": "function", "function": {"name": "explode", "arguments": "{}"}}])
    db.append_message("s_boom", role="tool", content="sensitive rows", tool_name="explode", tool_call_id="call_x")
    visibility(Provider(RedactingRule(visible={"s_boom"}, shown_tools={"explode"})))
    assert "sensitive rows" not in session_search(session_id="s_boom", db=db)


def _write_plugin(home, name, body):
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": name, "version": "0.1.0", "description": name}))
    (plugin_dir / "__init__.py").write_text(body)


_PROVIDER_PLUGIN = (
    "from agent.session_visibility import SessionJudgement, SessionVisibility\n\n"
    "class Nobody(SessionJudgement):\n"
    "    def may_see(self, place):\n"
    "        return False\n\n"
    "class Provider(SessionVisibility):\n"
    "    name = '{name}'\n"
    "    def judgement(self):\n"
    "        return Nobody()\n\n"
    "def register(ctx):\n"
    "    ctx.register_session_visibility(Provider())\n"
)


def test_a_plugin_registers_the_one_session_visibility(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_plugin(home, "first-visibility", _PROVIDER_PLUGIN.format(name="first"))
    _write_plugin(home, "second-visibility", _PROVIDER_PLUGIN.format(name="second"))
    _write_plugin(home, "wrong-visibility",
                  "def register(ctx):\n    ctx.register_session_visibility(object())\n")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["first-visibility", "second-visibility", "wrong-visibility"]}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.plugins as plugins_mod
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)
    provider = plugins_mod.get_plugin_session_visibility()
    assert isinstance(provider, SessionVisibility) and provider.name in {"first", "second"}
