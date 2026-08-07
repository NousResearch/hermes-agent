"""Per-call ``user_id`` on the Camofox backend (issue #77273).

Camofox maps each ``userId`` to its own browser profile, so passing a distinct
one per call lets a single Hermes process drive several signed-in accounts.
These tests pin the resolution precedence, the session isolation contract, the
validation floor on a model-supplied identity, and the Camofox-gated schema.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_camofox import (
    InvalidCamofoxUserId,
    _get_session,
    _validate_call_user_id,
    camofox_click,
    camofox_close,
    camofox_navigate,
    camofox_snapshot,
    list_camofox_sessions,
)
from tools.browser_camofox_state import get_camofox_identity


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def _tab_response(tab_id="tab-1", url="https://example.com"):
    return _mock_response(json_data={"tabId": tab_id, "url": url})


@pytest.fixture(autouse=True)
def _clear_session_state():
    import tools.browser_camofox as mod
    yield
    with mod._sessions_lock:
        mod._sessions.clear()
    mod._vnc_url = None
    mod._vnc_url_checked = False


@pytest.fixture
def camofox_env(tmp_path, monkeypatch):
    """A clean Camofox-enabled environment with no configured identity."""
    import tools.browser_camofox as mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    for var in ("CAMOFOX_USER_ID", "CAMOFOX_SESSION_KEY", "CAMOFOX_ADOPT_EXISTING_TAB",
                "BROWSER_CDP_URL"):
        monkeypatch.delenv(var, raising=False)
    # Mark the VNC probe as already done (no VNC). camofox_navigate calls
    # get_vnc_url(), which would otherwise issue a real GET /health — unrelated
    # to identity routing, and a network round-trip in every test.
    mod._vnc_url = None
    mod._vnc_url_checked = True
    return tmp_path


# ---------------------------------------------------------------------------
# Identity resolution precedence
# ---------------------------------------------------------------------------


class TestResolutionPrecedence:
    """per-call user_id > CAMOFOX_USER_ID / config > managed profile > random."""

    def test_per_call_beats_env_identity(self, camofox_env, monkeypatch):
        monkeypatch.setenv("CAMOFOX_USER_ID", "env-identity")

        session = _get_session("task-1", user_id="acct-alice")

        assert session["user_id"] == "acct-alice"
        assert session["managed"] is True

    def test_per_call_beats_config_identity(self, camofox_env):
        config = {"browser": {"camofox": {"user_id": "config-identity"}}}

        with patch("tools.browser_camofox.load_config", return_value=config):
            session = _get_session("task-1", user_id="acct-alice")

        assert session["user_id"] == "acct-alice"

    def test_per_call_beats_managed_persistence(self, camofox_env):
        config = {"browser": {"camofox": {"managed_persistence": True}}}

        with patch("tools.browser_camofox.load_config", return_value=config):
            session = _get_session("task-1", user_id="acct-alice")
            profile_identity = get_camofox_identity("task-1")

        assert session["user_id"] == "acct-alice"
        assert session["user_id"] != profile_identity["user_id"]

    def test_env_identity_still_wins_when_no_per_call_id(self, camofox_env, monkeypatch):
        """Omitting user_id must not disturb the existing precedence chain."""
        monkeypatch.setenv("CAMOFOX_USER_ID", "env-identity")

        session = _get_session("task-1")

        assert session["user_id"] == "env-identity"

    def test_omitting_user_id_keeps_ephemeral_behaviour(self, camofox_env):
        session = _get_session("task-1")

        assert session["user_id"].startswith("hermes_")
        assert session["managed"] is False
        assert session["session_key"] == "task_task-1"

    def test_default_session_still_keyed_by_bare_task_id(self, camofox_env):
        """The no-user_id key space must stay byte-identical to pre-#77273."""
        import tools.browser_camofox as mod

        _get_session("task-1")

        with mod._sessions_lock:
            assert "task-1" in mod._sessions


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    def test_two_identities_get_independent_sessions(self, camofox_env):
        alice = _get_session("task-1", user_id="acct-alice")
        bob = _get_session("task-1", user_id="acct-bob")

        assert alice is not bob
        assert alice["user_id"] == "acct-alice"
        assert bob["user_id"] == "acct-bob"

    def test_second_identity_does_not_evict_the_first(self, camofox_env):
        alice = _get_session("task-1", user_id="acct-alice")
        alice["tab_id"] = "tab-alice"
        _get_session("task-1", user_id="acct-bob")

        assert _get_session("task-1", user_id="acct-alice")["tab_id"] == "tab-alice"

    def test_same_identity_reuses_the_session(self, camofox_env):
        first = _get_session("task-1", user_id="acct-alice")
        second = _get_session("task-1", user_id="acct-alice")

        assert first is second

    def test_per_call_identity_does_not_disturb_default_session(self, camofox_env):
        default = _get_session("task-1")
        scoped = _get_session("task-1", user_id="acct-alice")

        assert default["user_id"] != scoped["user_id"]
        assert _get_session("task-1") is default

    def test_navigate_sends_requested_user_id(self, camofox_env):
        with patch("tools.browser_camofox.requests.post",
                   return_value=_tab_response()) as mock_post:
            result = json.loads(
                camofox_navigate("https://example.com", task_id="t1", user_id="acct-alice")
            )

        assert result["success"] is True
        assert mock_post.call_args.kwargs["json"]["userId"] == "acct-alice"

    def test_repeat_navigate_same_identity_reuses_tab(self, camofox_env):
        posts = []

        def _capture(url, json=None, timeout=None, headers=None):
            posts.append((url, json))
            return _tab_response(tab_id="tab-alice")

        with patch("tools.browser_camofox.requests.post", side_effect=_capture):
            camofox_navigate("https://example.com", task_id="t1", user_id="acct-alice")
            camofox_navigate("https://example.com/two", task_id="t1", user_id="acct-alice")

        # First call creates the tab; the second navigates the existing one.
        assert posts[0][0].endswith("/tabs")
        assert posts[1][0].endswith("/tabs/tab-alice/navigate")
        assert posts[1][1]["userId"] == "acct-alice"

    def test_two_identities_create_two_tabs(self, camofox_env):
        created = []

        def _capture(url, json=None, timeout=None, headers=None):
            created.append(json["userId"])
            return _tab_response(tab_id=f"tab-{len(created)}")

        with patch("tools.browser_camofox.requests.post", side_effect=_capture):
            camofox_navigate("https://example.com", task_id="t1", user_id="acct-alice")
            camofox_navigate("https://example.com", task_id="t1", user_id="acct-bob")

        assert created == ["acct-alice", "acct-bob"]


# ---------------------------------------------------------------------------
# Validation of a model-supplied identity
# ---------------------------------------------------------------------------


class TestUserIdValidation:
    """A per-call user_id reaches DELETE /sessions/<id> as a path segment."""

    @pytest.mark.parametrize("value", [
        "acct-alice", "acct_bob", "a", "user.name", "ns:account", "user@example.com",
        "A1", "x" * 64,
    ])
    def test_accepts_reasonable_identities(self, value):
        assert _validate_call_user_id(value) == value

    @pytest.mark.parametrize("value", [
        "../evil",          # path traversal
        "..",               # DELETE /sessions/.. can normalize to DELETE /sessions
        ".",
        "a/b",              # path separator
        "a%2Fb",            # percent-encoded separator
        "a\\b",
        "-leading-dash",
        ".leading-dot",
        "has space",
        "new\nline",
        "",
        "   ",
        "x" * 65,           # over the length cap
    ])
    def test_rejects_unsafe_identities(self, value):
        with pytest.raises(InvalidCamofoxUserId):
            _validate_call_user_id(value)

    @pytest.mark.parametrize("value", ["acct\n", "acct\n\n", "acct\r\n", "acct-a\n"])
    def test_charset_rejects_trailing_newline_on_its_own(self, value):
        """Python's ``$`` matches before a trailing newline; fullmatch does not.

        The charset is a security boundary (the value becomes a URL path
        segment), so it must hold without depending on the .strip() that
        happens to run first. Surrounding whitespace is still normalized away
        by the caller, so the validator itself accepts these and returns the
        trimmed value — the point is that the *pattern* would not.
        """
        import tools.browser_camofox as mod

        assert mod._CALL_USER_ID_RE.fullmatch(value) is None
        assert _validate_call_user_id(value) == value.strip()

    @pytest.mark.parametrize("value", ["ac\nct", "acct\nevil", "a\rb"])
    def test_rejects_interior_newlines(self, value):
        with pytest.raises(InvalidCamofoxUserId):
            _validate_call_user_id(value)

    @pytest.mark.parametrize("value", [0, False, 12345, 1.5, [], {}, ["acct-alice"]])
    def test_rejects_non_strings_instead_of_falling_back(self, value):
        """A type-confused argument must error, never select another account.

        ``0``/``False`` are falsy: a truthiness gate would skip validation and
        silently run under the *default* identity while the caller believes it
        addressed a named account — the exact cross-account action the
        parameter exists to prevent.
        """
        with pytest.raises(InvalidCamofoxUserId):
            _validate_call_user_id(value)

    @pytest.mark.parametrize("value", [0, False, 12345, []])
    def test_non_string_is_rejected_end_to_end(self, camofox_env, value):
        import tools.browser_camofox as mod

        with patch("tools.browser_camofox.requests.post") as mock_post:
            result = json.loads(
                camofox_navigate("https://example.com", task_id="t1", user_id=value)
            )

        assert result["success"] is False
        assert "expected a string" in result["error"]
        mock_post.assert_not_called()
        with mod._sessions_lock:
            assert mod._sessions == {}

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_absent_identity_uses_default_session(self, camofox_env, value):
        """None and empty strings mean "not supplied", not "invalid"."""
        import tools.browser_camofox as mod

        session = _get_session("task-1", user_id=value)

        assert session["user_id"].startswith("hermes_")
        with mod._sessions_lock:
            assert list(mod._sessions) == ["task-1"]

    def test_invalid_identity_short_circuits_before_any_request(self, camofox_env):
        with (
            patch("tools.browser_camofox.requests.post") as mock_post,
            patch("tools.browser_camofox.requests.get") as mock_get,
            patch("tools.browser_camofox.requests.delete") as mock_delete,
        ):
            result = json.loads(
                camofox_navigate("https://example.com", task_id="t1", user_id="../evil")
            )

        assert result["success"] is False
        assert "user_id" in result["error"]
        mock_post.assert_not_called()
        mock_get.assert_not_called()
        mock_delete.assert_not_called()

    def test_invalid_identity_rejected_by_close(self, camofox_env):
        with patch("tools.browser_camofox.requests.delete") as mock_delete:
            result = json.loads(camofox_close("t1", user_id=".."))

        assert result["success"] is False
        mock_delete.assert_not_called()

    def test_invalid_identity_rejected_by_console(self, camofox_env):
        """camofox_console has no session to fall back on; it must still error."""
        from tools.browser_camofox import camofox_console

        result = json.loads(camofox_console(user_id=12345))

        assert result["success"] is False
        assert "expected a string" in result["error"]

    def test_invalid_identity_creates_no_session(self, camofox_env):
        import tools.browser_camofox as mod

        with pytest.raises(InvalidCamofoxUserId):
            _get_session("task-1", user_id="../evil")

        with mod._sessions_lock:
            assert mod._sessions == {}

    def test_operator_configured_identity_bypasses_call_validation(self, camofox_env, monkeypatch):
        """Config/env identities are operator-owned and predate the charset rule."""
        monkeypatch.setenv("CAMOFOX_USER_ID", "legacy identity/with weird chars")

        session = _get_session("task-1")

        assert session["user_id"] == "legacy identity/with weird chars"


class TestConsoleDoesNotRubberStampIdentity:
    """camofox_console never opens a session, so it must not echo an identity."""

    def test_does_not_echo_unresolved_identity(self, camofox_env):
        from tools.browser_camofox import camofox_console

        result = json.loads(camofox_console(user_id="acct-never-created"))

        assert result["success"] is True
        assert "user_id" not in result

    def test_does_not_create_a_session(self, camofox_env):
        import tools.browser_camofox as mod
        from tools.browser_camofox import camofox_console

        camofox_console(user_id="acct-alice")

        with mod._sessions_lock:
            assert mod._sessions == {}


class TestErrorsDiscloseIdentity:
    """Echoing only on success can't distinguish failure from acting as someone else."""

    def test_missing_tab_error_echoes_identity(self, camofox_env):
        _get_session("task-1", user_id="acct-alice")

        result = json.loads(camofox_click("@e1", task_id="task-1", user_id="acct-alice"))

        assert result["success"] is False
        assert result["user_id"] == "acct-alice"

    def test_http_failure_echoes_identity(self, camofox_env):
        session = _get_session("task-1", user_id="acct-alice")
        session["tab_id"] = "tab-alice"

        with (
            patch("tools.browser_camofox._camofox_private_page_block", return_value=None),
            patch("tools.browser_camofox._post", side_effect=RuntimeError("boom")),
        ):
            result = json.loads(camofox_click("@e1", task_id="task-1", user_id="acct-alice"))

        assert result["success"] is False
        assert result["user_id"] == "acct-alice"

    def test_private_page_block_echoes_identity(self, camofox_env):
        import tools.browser_camofox as mod

        session = _get_session("task-1", user_id="acct-alice")
        session["tab_id"] = "tab-alice"

        with (
            patch("tools.browser_tool._eval_ssrf_guard_active", return_value=True),
            patch("tools.browser_tool._camofox_current_page_private_url",
                  return_value="http://169.254.169.254/"),
        ):
            blocked = json.loads(
                mod._camofox_private_page_block(session, "task-1", "click")
            )

        assert blocked["success"] is False
        assert blocked["user_id"] == "acct-alice"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_does_not_destroy_per_call_profile(self, camofox_env):
        """DELETE /sessions/<id> wipes all of a user's data — never for a
        caller-owned identity."""
        _get_session("task-1", user_id="acct-alice")

        with patch("tools.browser_camofox.requests.delete",
                   return_value=_mock_response()) as mock_delete:
            result = json.loads(camofox_close("task-1", user_id="acct-alice"))

        assert result["closed"] is True
        assert not [c for c in mock_delete.call_args_list if "/sessions/" in c.args[0]]

    def test_close_reclaims_the_per_call_tab(self, camofox_env):
        """Closing the tab is what keeps each new identity from leaking a window.

        Camofox's POST /pressure/cleanup is caller-triggered, not an automatic
        reaper, so a tab Hermes opened and never closed stays open.
        """
        session = _get_session("task-1", user_id="acct-alice")
        session["tab_id"] = "tab-alice"

        with patch("tools.browser_camofox.requests.delete",
                   return_value=_mock_response()) as mock_delete:
            camofox_close("task-1", user_id="acct-alice")

        mock_delete.assert_called_once()
        assert mock_delete.call_args.args[0].endswith("/tabs/tab-alice")
        # userId is a query parameter on this endpoint, not a path segment.
        assert mock_delete.call_args.kwargs["params"] == {"userId": "acct-alice"}

    def test_close_skips_tab_delete_when_no_tab_was_opened(self, camofox_env):
        _get_session("task-1", user_id="acct-alice")

        with patch("tools.browser_camofox.requests.delete") as mock_delete:
            camofox_close("task-1", user_id="acct-alice")

        mock_delete.assert_not_called()

    def test_close_still_destroys_ephemeral_default_session(self, camofox_env):
        """Regression guard: the pre-existing teardown path is unchanged."""
        _get_session("task-1")

        with patch("tools.browser_camofox.requests.delete",
                   return_value=_mock_response()) as mock_delete:
            camofox_close("task-1")

        mock_delete.assert_called_once()
        assert "/sessions/hermes_" in mock_delete.call_args.args[0]

    def test_mixed_teardown_uses_the_right_verb_per_identity(self, camofox_env):
        default = _get_session("task-1")
        default["tab_id"] = "tab-default"
        alice = _get_session("task-1", user_id="acct-alice")
        alice["tab_id"] = "tab-alice"

        with patch("tools.browser_camofox.requests.delete",
                   return_value=_mock_response()) as mock_delete:
            camofox_close("task-1")

        targets = sorted(c.args[0].rsplit("/", 2)[-2:] for c in mock_delete.call_args_list)
        assert targets == [["sessions", default["user_id"]], ["tabs", "tab-alice"]]

    def test_one_failing_delete_does_not_strand_the_others(self, camofox_env):
        """_drop_sessions already removed every entry, so a skipped sibling
        could never be reclaimed by anyone."""
        for name in ("acct-a", "acct-b", "acct-c"):
            _get_session("task-1", user_id=name)["tab_id"] = f"tab-{name}"

        attempted = []

        def _flaky(url, json=None, params=None, timeout=None, headers=None):
            attempted.append(params["userId"])
            if params["userId"] == "acct-a":
                raise RuntimeError("camofox 500")
            return _mock_response()

        with patch("tools.browser_camofox.requests.delete", side_effect=_flaky):
            result = json.loads(camofox_close("task-1"))

        assert attempted == ["acct-a", "acct-b", "acct-c"]
        assert "acct-a" in result["warning"]
        assert result["closed"] is True

    def test_close_without_user_id_reaps_every_identity(self, camofox_env):
        """Task teardown must not leak per-user sessions."""
        import tools.browser_camofox as mod

        _get_session("task-1")
        _get_session("task-1", user_id="acct-alice")
        _get_session("task-1", user_id="acct-bob")

        with patch("tools.browser_camofox.requests.delete", return_value=_mock_response()):
            camofox_close("task-1")

        with mod._sessions_lock:
            assert mod._sessions == {}

    def test_close_leaves_other_tasks_alone(self, camofox_env):
        import tools.browser_camofox as mod

        _get_session("task-1", user_id="acct-alice")
        _get_session("task-2", user_id="acct-alice")

        with patch("tools.browser_camofox.requests.delete", return_value=_mock_response()):
            camofox_close("task-1")

        assert [s["task_id"] for s in list_camofox_sessions()] == ["task-2"]
        with mod._sessions_lock:
            assert len(mod._sessions) == 1

    def test_soft_cleanup_reaps_per_user_sessions(self, camofox_env):
        import tools.browser_camofox as mod
        from tools.browser_camofox import camofox_soft_cleanup

        config = {"browser": {"camofox": {"managed_persistence": True}}}
        with patch("tools.browser_camofox.load_config", return_value=config):
            _get_session("task-1")
            _get_session("task-1", user_id="acct-alice")

            with patch("tools.browser_camofox.requests.delete") as mock_delete:
                assert camofox_soft_cleanup("task-1") is True

        # No tab was opened for either session, so nothing to release; the
        # managed profile must not be touched regardless.
        assert not [c for c in mock_delete.call_args_list if "/sessions/" in c.args[0]]
        with mod._sessions_lock:
            assert mod._sessions == {}

    def test_soft_cleanup_closes_per_call_tabs_but_spares_managed_profile(self, camofox_env):
        from tools.browser_camofox import camofox_soft_cleanup

        config = {"browser": {"camofox": {"managed_persistence": True}}}
        with patch("tools.browser_camofox.load_config", return_value=config):
            _get_session("task-1")["tab_id"] = "tab-managed"
            _get_session("task-1", user_id="acct-alice")["tab_id"] = "tab-alice"

            with patch("tools.browser_camofox.requests.delete",
                       return_value=_mock_response()) as mock_delete:
                camofox_soft_cleanup("task-1")

        # Only the per-call tab is closed. The managed profile keeps its tab
        # AND its cookies — that is the point of soft cleanup.
        mock_delete.assert_called_once()
        assert mock_delete.call_args.args[0].endswith("/tabs/tab-alice")


# ---------------------------------------------------------------------------
# Tab adoption follows the existing operator opt-in
# ---------------------------------------------------------------------------


class TestTabAdoption:
    def test_adoption_off_by_default(self, camofox_env):
        with patch("tools.browser_camofox._get") as mock_get:
            session = _get_session("task-1", user_id="acct-alice")

        assert session["adopt_existing_tab"] is False
        mock_get.assert_not_called()

    def test_adoption_honours_config_flag(self, camofox_env, monkeypatch):
        monkeypatch.setenv("CAMOFOX_ADOPT_EXISTING_TAB", "true")

        with patch("tools.browser_camofox._get",
                   return_value={"tabs": [{"tabId": "existing-tab", "listItemId": "other"}]}) as mock_get:
            session = _get_session("task-1", user_id="acct-alice")

        assert session["adopt_existing_tab"] is True
        assert session["tab_id"] == "existing-tab"
        mock_get.assert_called_once_with(
            "/tabs", params={"userId": "acct-alice"}, timeout=5
        )


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


class TestResultEchoesIdentity:
    def test_navigate_echoes_user_id(self, camofox_env):
        with patch("tools.browser_camofox.requests.post", return_value=_tab_response()):
            result = json.loads(
                camofox_navigate("https://example.com", task_id="t1", user_id="acct-alice")
            )

        assert result["user_id"] == "acct-alice"

    def test_navigate_echoes_resolved_identity_without_per_call_id(self, camofox_env):
        with patch("tools.browser_camofox.requests.post", return_value=_tab_response()):
            result = json.loads(camofox_navigate("https://example.com", task_id="t1"))

        assert result["user_id"].startswith("hermes_")

    def test_click_echoes_user_id(self, camofox_env):
        session = _get_session("task-1", user_id="acct-alice")
        session["tab_id"] = "tab-alice"

        with (
            patch("tools.browser_camofox._camofox_private_page_block", return_value=None),
            patch("tools.browser_camofox._post", return_value={"url": "https://example.com"}),
        ):
            result = json.loads(camofox_click("@e5", task_id="task-1", user_id="acct-alice"))

        assert result["user_id"] == "acct-alice"

    def test_snapshot_echoes_user_id(self, camofox_env):
        session = _get_session("task-1", user_id="acct-alice")
        session["tab_id"] = "tab-alice"

        with (
            patch("tools.browser_camofox._camofox_private_page_block", return_value=None),
            patch("tools.browser_camofox._get",
                  return_value={"snapshot": "- button", "refsCount": 1}),
        ):
            result = json.loads(camofox_snapshot(task_id="task-1", user_id="acct-alice"))

        assert result["user_id"] == "acct-alice"


class TestListSessions:
    def test_lists_every_active_identity(self, camofox_env):
        _get_session("task-1")
        _get_session("task-1", user_id="acct-alice")
        _get_session("task-2", user_id="acct-bob")

        sessions = {(s["task_id"], s["user_id"]): s for s in list_camofox_sessions()}

        assert len(sessions) == 3
        assert sessions[("task-1", "acct-alice")]["explicit_user_id"] is True
        assert sessions[("task-2", "acct-bob")]["explicit_user_id"] is True
        default = next(s for s in sessions.values() if not s["explicit_user_id"])
        assert default["task_id"] == "task-1"

    def test_reports_tab_ids(self, camofox_env):
        with patch("tools.browser_camofox.requests.post",
                   return_value=_tab_response(tab_id="tab-alice")):
            camofox_navigate("https://example.com", task_id="t1", user_id="acct-alice")

        [session] = list_camofox_sessions()
        assert session["tab_id"] == "tab-alice"
        assert session["user_id"] == "acct-alice"

    def test_empty_when_no_sessions(self):
        assert list_camofox_sessions() == []


# ---------------------------------------------------------------------------
# Model-facing schema is Camofox-gated
# ---------------------------------------------------------------------------


_CAMOFOX_BACKED_TOOLS = [
    "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
    "browser_scroll", "browser_back", "browser_press", "browser_console",
    "browser_get_images", "browser_vision",
]


def _resolved_schema(tool_name):
    """Schema the model would see, with dynamic overrides applied."""
    from tools.registry import registry

    entry = registry._tools[tool_name]
    schema = {**entry.schema}
    schema.update(entry.dynamic_schema_overrides() or {})
    return schema


@pytest.fixture
def _no_cdp_override(monkeypatch):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    with patch("tools.browser_camofox._config_cdp_url", return_value=""):
        yield


class TestSchemaGating:
    """user_id costs nothing on backends that cannot honour it."""

    @pytest.mark.parametrize("tool_name", _CAMOFOX_BACKED_TOOLS)
    def test_absent_without_camofox(self, tool_name, monkeypatch, _no_cdp_override):
        monkeypatch.delenv("CAMOFOX_URL", raising=False)

        properties = _resolved_schema(tool_name)["parameters"]["properties"]

        assert "user_id" not in properties

    @pytest.mark.parametrize("tool_name", _CAMOFOX_BACKED_TOOLS)
    def test_present_with_camofox(self, tool_name, monkeypatch, _no_cdp_override):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        properties = _resolved_schema(tool_name)["parameters"]["properties"]

        assert properties["user_id"]["type"] == "string"

    @pytest.mark.parametrize("tool_name", _CAMOFOX_BACKED_TOOLS)
    def test_only_user_id_changes(self, tool_name, monkeypatch, _no_cdp_override):
        """The override must not disturb required fields or descriptions."""
        monkeypatch.delenv("CAMOFOX_URL", raising=False)
        without = _resolved_schema(tool_name)
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        with_camofox = _resolved_schema(tool_name)

        assert without["description"] == with_camofox["description"]
        assert without["parameters"].get("required") == with_camofox["parameters"].get("required")
        assert (
            set(with_camofox["parameters"]["properties"])
            - set(without["parameters"]["properties"])
            == {"user_id"}
        )

    def test_static_schema_is_not_mutated(self, monkeypatch, _no_cdp_override):
        """Building the override twice must not leak user_id into the base."""
        import tools.browser_tool as bt

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        _resolved_schema("browser_navigate")
        _resolved_schema("browser_navigate")

        static_props = bt._BROWSER_SCHEMA_MAP["browser_navigate"]["parameters"]["properties"]
        assert "user_id" not in static_props

    def test_handler_forwards_user_id(self, monkeypatch, _no_cdp_override):
        from tools.registry import registry

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        seen = {}

        def _fake_navigate(url, task_id=None, user_id=None):
            seen.update(url=url, task_id=task_id, user_id=user_id)
            return json.dumps({"success": True})

        with patch("tools.browser_tool.browser_navigate", _fake_navigate):
            registry._tools["browser_navigate"].handler(
                {"url": "https://example.com", "user_id": "acct-alice"}, task_id="t1"
            )

        assert seen == {
            "url": "https://example.com", "task_id": "t1", "user_id": "acct-alice",
        }

    def test_description_does_not_invite_omitting_mid_sequence(self, monkeypatch, _no_cdp_override):
        """Refs are not identity-scoped, so 'omit to use the default session'
        would invite applying one account's refs to another profile's page."""
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        description = (
            _resolved_schema("browser_navigate")["parameters"]["properties"]["user_id"]
            ["description"].lower()
        )

        assert "does not continue" in description
        assert "every call" in description


class TestBackendMismatchIsRejected:
    """A backend that changed underneath the model must fail loud, not silently
    run the call on a shared browser signed in as someone else."""

    @pytest.fixture
    def cdp_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222/devtools/x")
        import tools.browser_tool as bt

        assert bt._is_camofox_mode() is False
        return bt

    @pytest.mark.parametrize(("attr", "kwargs"), [
        ("browser_navigate", {"url": "https://example.com"}),
        ("browser_snapshot", {}),
        ("browser_click", {"ref": "@e1"}),
        ("browser_type", {"ref": "@e1", "text": "bob-password"}),
        ("browser_scroll", {"direction": "down"}),
        ("browser_back", {}),
        ("browser_press", {"key": "Enter"}),
        ("browser_console", {}),
        ("browser_get_images", {}),
        ("browser_vision", {"question": "what is here?"}),
    ])
    def test_rejects_user_id_without_running_the_call(self, cdp_backend, attr, kwargs):
        with (
            patch.object(cdp_backend, "_is_safe_url", lambda url: True),
            patch.object(cdp_backend, "_is_always_blocked_url", lambda url: False),
            patch.object(cdp_backend, "check_website_access", lambda url: None),
            patch.object(cdp_backend, "_run_browser_command") as mock_cmd,
        ):
            result = json.loads(
                getattr(cdp_backend, attr)(task_id="t1", user_id="acct-bob", **kwargs)
            )

        assert result["success"] is False
        assert "Camofox backend" in result["error"]
        mock_cmd.assert_not_called()

    def test_eval_path_rejects_user_id(self, cdp_backend):
        with patch.object(cdp_backend, "_run_browser_command") as mock_cmd:
            result = json.loads(
                cdp_backend.browser_console(
                    expression="document.title", task_id="t1", user_id="acct-bob"
                )
            )

        assert result["success"] is False
        assert "Camofox backend" in result["error"]
        mock_cmd.assert_not_called()

    def test_calls_without_user_id_are_unaffected(self, cdp_backend):
        with (
            patch.object(cdp_backend, "_is_safe_url", lambda url: True),
            patch.object(cdp_backend, "_is_always_blocked_url", lambda url: False),
            patch.object(cdp_backend, "check_website_access", lambda url: None),
            patch.object(cdp_backend, "_run_browser_command",
                         return_value={"success": True, "data": {}}) as mock_cmd,
        ):
            result = json.loads(cdp_backend.browser_click(ref="@e1", task_id="t1"))

        assert result["success"] is True
        assert mock_cmd.called


class TestEvalErrorClassification:
    """_camofox_eval degrades gracefully on 404/405/501 by substring-matching
    the error text, which must not swallow a validation failure."""

    def test_identity_error_is_not_reported_as_unsupported_server(self, camofox_env):
        import tools.browser_tool as bt

        result = json.loads(
            bt._camofox_eval("document.title", task_id="t1", user_id="team/404-ops")
        )

        assert result["success"] is False
        assert "Invalid user_id" in result["error"]
        assert "not supported by this Camofox server" not in result["error"]

    def test_genuine_unsupported_server_still_degrades(self, camofox_env):
        import tools.browser_tool as bt

        with (
            patch("tools.browser_camofox._ensure_tab",
                  return_value={"tab_id": "tab-1", "user_id": "acct-alice"}),
            patch("tools.browser_camofox._post", side_effect=RuntimeError("404 Not Found")),
        ):
            result = json.loads(
                bt._camofox_eval("document.title", task_id="t1", user_id="acct-alice")
            )

        assert result["success"] is False
        assert "not supported by this Camofox server" in result["error"]

    def test_eval_result_echoes_identity(self, camofox_env):
        import tools.browser_tool as bt

        with (
            patch("tools.browser_camofox._ensure_tab",
                  return_value={"tab_id": "tab-1", "user_id": "acct-alice"}),
            patch("tools.browser_camofox._post", return_value={"result": "Example"}),
            patch("tools.browser_tool._eval_ssrf_guard_active", return_value=False),
        ):
            result = json.loads(
                bt._camofox_eval("document.title", task_id="t1", user_id="acct-alice")
            )

        assert result["success"] is True
        assert result["user_id"] == "acct-alice"
