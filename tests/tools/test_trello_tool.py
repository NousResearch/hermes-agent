"""Unit tests for tools/trello_tool.py.

Tests cover:
- check_requirements: missing vars, partial, both set
- trello_handler: credential guard, unknown action, missing required params
- list_cards special validation (needs list_id OR board_id)
- Each action implementation mocked via urllib.request.urlopen
- HTTP error handling (401, 403, 404, generic)
"""

import json
import sys
import types
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to mock urllib responses
# ---------------------------------------------------------------------------

class _MockHTTPResponse:
    """Minimal mock for the object returned by urllib.request.urlopen."""

    def __init__(self, payload: object, status: int = 200) -> None:
        self._data = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _mock_urlopen(payload, status=200):
    """Return a context-manager patch for urllib.request.urlopen."""
    return patch(
        "urllib.request.urlopen",
        return_value=_MockHTTPResponse(payload, status=status),
    )


def _mock_http_error(status: int, body: str = "error"):
    """Raise an HTTPError from urlopen."""
    import urllib.error

    err = urllib.error.HTTPError(
        url="https://api.trello.com/1/test",
        code=status,
        msg=body,
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(body.encode()),
    )
    return patch("urllib.request.urlopen", side_effect=err)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure Trello env vars are absent before each test."""
    monkeypatch.delenv("TRELLO_API_KEY", raising=False)
    monkeypatch.delenv("TRELLO_API_TOKEN", raising=False)


@pytest.fixture()
def with_creds(monkeypatch):
    """Set both credentials."""
    monkeypatch.setenv("TRELLO_API_KEY", "test-api-key")
    monkeypatch.setenv("TRELLO_API_TOKEN", "test-api-token")


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

# We import at the module level so the registry.register() call fires once.
from tools import trello_tool as tt  # noqa: E402


# ---------------------------------------------------------------------------
# check_trello_requirements
# ---------------------------------------------------------------------------

class TestCheckRequirements:
    def test_no_env_vars(self, monkeypatch):
        assert tt.check_trello_requirements() is False

    def test_only_key(self, monkeypatch):
        monkeypatch.setenv("TRELLO_API_KEY", "k")
        assert tt.check_trello_requirements() is False

    def test_only_token(self, monkeypatch):
        monkeypatch.setenv("TRELLO_API_TOKEN", "t")
        assert tt.check_trello_requirements() is False

    def test_both_empty_strings(self, monkeypatch):
        monkeypatch.setenv("TRELLO_API_KEY", "   ")
        monkeypatch.setenv("TRELLO_API_TOKEN", "  ")
        assert tt.check_trello_requirements() is False

    def test_both_set(self, monkeypatch):
        monkeypatch.setenv("TRELLO_API_KEY", "key123")
        monkeypatch.setenv("TRELLO_API_TOKEN", "tok456")
        assert tt.check_trello_requirements() is True


# ---------------------------------------------------------------------------
# Handler — credential guard
# ---------------------------------------------------------------------------

class TestHandlerCredentialGuard:
    def test_no_credentials_returns_error(self):
        result = json.loads(tt.trello_handler(action="list_boards"))
        assert "error" in result
        assert "TRELLO_API_KEY" in result["error"]

    def test_unknown_action(self, with_creds):
        result = json.loads(tt.trello_handler(action="does_not_exist"))
        assert "error" in result
        assert "Unknown action" in result["error"]


# ---------------------------------------------------------------------------
# Handler — missing required params
# ---------------------------------------------------------------------------

class TestMissingParams:
    def test_get_board_missing_board_id(self, with_creds):
        result = json.loads(tt.trello_handler(action="get_board"))
        assert "error" in result
        assert "board_id" in result["error"]

    def test_get_card_missing_card_id(self, with_creds):
        result = json.loads(tt.trello_handler(action="get_card"))
        assert "error" in result
        assert "card_id" in result["error"]

    def test_create_card_missing_list_id(self, with_creds):
        result = json.loads(tt.trello_handler(action="create_card", name="Test"))
        assert "error" in result
        assert "list_id" in result["error"]

    def test_create_card_missing_name(self, with_creds):
        result = json.loads(tt.trello_handler(action="create_card", list_id="abc"))
        assert "error" in result
        assert "name" in result["error"]

    def test_list_cards_no_list_or_board(self, with_creds):
        result = json.loads(tt.trello_handler(action="list_cards"))
        assert "error" in result
        assert "list_id" in result["error"] or "board_id" in result["error"]

    def test_search_missing_query(self, with_creds):
        result = json.loads(tt.trello_handler(action="search"))
        assert "error" in result
        assert "query" in result["error"]

    def test_add_comment_missing_text(self, with_creds):
        result = json.loads(tt.trello_handler(action="add_comment", card_id="card1"))
        assert "error" in result
        assert "text" in result["error"]


# ---------------------------------------------------------------------------
# Action — list_boards
# ---------------------------------------------------------------------------

class TestListBoards:
    def test_success(self, with_creds):
        fake_boards = [
            {"id": "b1", "name": "Board One", "desc": "desc", "url": "https://trello.com/b1", "closed": False, "idOrganization": None},
        ]
        with _mock_urlopen(fake_boards):
            result = json.loads(tt.trello_handler(action="list_boards"))
        assert result["count"] == 1
        assert result["boards"][0]["id"] == "b1"
        assert result["boards"][0]["name"] == "Board One"


# ---------------------------------------------------------------------------
# Action — get_board
# ---------------------------------------------------------------------------

class TestGetBoard:
    def test_success(self, with_creds):
        fake_board = {
            "id": "b1", "name": "Board One", "desc": "a desc",
            "url": "https://trello.com/b1", "closed": False,
            "idOrganization": "org1", "dateLastActivity": "2024-01-01T00:00:00.000Z",
            "prefs": {"background": "blue", "permissionLevel": "private"},
        }
        with _mock_urlopen(fake_board):
            result = json.loads(tt.trello_handler(action="get_board", board_id="b1"))
        assert result["id"] == "b1"
        assert result["prefs"]["background"] == "blue"


# ---------------------------------------------------------------------------
# Action — list_lists
# ---------------------------------------------------------------------------

class TestListLists:
    def test_success(self, with_creds):
        fake_lists = [
            {"id": "l1", "name": "To Do", "closed": False, "pos": 1},
            {"id": "l2", "name": "Done", "closed": False, "pos": 2},
        ]
        with _mock_urlopen(fake_lists):
            result = json.loads(tt.trello_handler(action="list_lists", board_id="b1"))
        assert result["count"] == 2
        assert result["lists"][0]["name"] == "To Do"


# ---------------------------------------------------------------------------
# Action — list_cards (by list_id and by board_id)
# ---------------------------------------------------------------------------

class TestListCards:
    _fake_cards = [
        {
            "id": "c1", "name": "Card One", "desc": "",
            "due": None, "dueComplete": False, "closed": False,
            "idList": "l1", "idBoard": "b1",
            "labels": [], "idMembers": [],
            "shortUrl": "https://trello.com/c/c1",
        }
    ]

    def test_by_list_id(self, with_creds):
        with _mock_urlopen(self._fake_cards):
            result = json.loads(tt.trello_handler(action="list_cards", list_id="l1"))
        assert result["count"] == 1
        assert result["cards"][0]["id"] == "c1"

    def test_by_board_id(self, with_creds):
        with _mock_urlopen(self._fake_cards):
            result = json.loads(tt.trello_handler(action="list_cards", board_id="b1"))
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# Action — get_card
# ---------------------------------------------------------------------------

class TestGetCard:
    def test_success(self, with_creds):
        fake_card = {
            "id": "c1", "name": "Card One", "desc": "some desc",
            "due": "2024-12-31T00:00:00.000Z", "dueComplete": False,
            "closed": False, "idList": "l1", "idBoard": "b1",
            "labels": [{"id": "lbl1", "name": "Bug", "color": "red"}],
            "idMembers": ["m1"],
            "members": [{"id": "m1", "fullName": "Alice", "username": "alice"}],
            "checklists": [
                {"id": "cl1", "name": "Tasks", "checkItems": [
                    {"name": "Step 1", "state": "complete"},
                    {"name": "Step 2", "state": "incomplete"},
                ]}
            ],
            "attachments": [],
            "shortUrl": "https://trello.com/c/c1",
            "dateLastActivity": "2024-01-01T00:00:00.000Z",
        }
        with _mock_urlopen(fake_card):
            result = json.loads(tt.trello_handler(action="get_card", card_id="c1"))
        assert result["id"] == "c1"
        assert result["labels"][0]["color"] == "red"
        assert len(result["checklists"]) == 1
        assert result["checklists"][0]["items"][0]["complete"] is True
        assert result["members"][0]["username"] == "alice"


# ---------------------------------------------------------------------------
# Action — create_card
# ---------------------------------------------------------------------------

class TestCreateCard:
    def test_success(self, with_creds):
        fake_response = {
            "id": "new_card", "name": "New Card",
            "idList": "l1", "idBoard": "b1",
            "shortUrl": "https://trello.com/c/new_card",
        }
        with _mock_urlopen(fake_response):
            result = json.loads(
                tt.trello_handler(action="create_card", list_id="l1", name="New Card")
            )
        assert result["success"] is True
        assert result["card_id"] == "new_card"

    def test_with_optional_fields(self, with_creds):
        fake_response = {
            "id": "c2", "name": "Card with opts",
            "idList": "l1", "idBoard": "b1",
            "shortUrl": "https://trello.com/c/c2",
        }
        with _mock_urlopen(fake_response):
            result = json.loads(
                tt.trello_handler(
                    action="create_card",
                    list_id="l1",
                    name="Card with opts",
                    desc="A description",
                    due="2024-12-31T00:00:00.000Z",
                    label_ids="lbl1,lbl2",
                )
            )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Action — update_card
# ---------------------------------------------------------------------------

class TestUpdateCard:
    def test_success(self, with_creds):
        fake_response = {"id": "c1", "name": "Updated Name", "closed": False, "idList": "l1"}
        with _mock_urlopen(fake_response):
            result = json.loads(
                tt.trello_handler(action="update_card", card_id="c1", name="Updated Name")
            )
        assert result["success"] is True
        assert result["name"] == "Updated Name"

    def test_no_fields_returns_error(self, with_creds):
        result = json.loads(tt.trello_handler(action="update_card", card_id="c1"))
        assert "error" in result
        assert "No fields" in result["error"]


# ---------------------------------------------------------------------------
# Action — move_card
# ---------------------------------------------------------------------------

class TestMoveCard:
    def test_success(self, with_creds):
        fake_response = {"id": "c1", "name": "Card", "idList": "l2", "idBoard": "b1"}
        with _mock_urlopen(fake_response):
            result = json.loads(
                tt.trello_handler(action="move_card", card_id="c1", list_id="l2")
            )
        assert result["success"] is True
        assert result["new_list_id"] == "l2"


# ---------------------------------------------------------------------------
# Action — archive_card
# ---------------------------------------------------------------------------

class TestArchiveCard:
    def test_success(self, with_creds):
        fake_response = {"id": "c1", "name": "Card", "closed": True}
        with _mock_urlopen(fake_response):
            result = json.loads(tt.trello_handler(action="archive_card", card_id="c1"))
        assert result["success"] is True
        assert result["closed"] is True


# ---------------------------------------------------------------------------
# Action — add_comment
# ---------------------------------------------------------------------------

class TestAddComment:
    def test_success(self, with_creds):
        fake_response = {"id": "act1"}
        with _mock_urlopen(fake_response):
            result = json.loads(
                tt.trello_handler(action="add_comment", card_id="c1", text="Hello!")
            )
        assert result["success"] is True
        assert result["text"] == "Hello!"


# ---------------------------------------------------------------------------
# Action — list_members
# ---------------------------------------------------------------------------

class TestListMembers:
    def test_success(self, with_creds):
        fake_members = [
            {"id": "m1", "fullName": "Alice", "username": "alice"},
            {"id": "m2", "fullName": "Bob", "username": "bob"},
        ]
        with _mock_urlopen(fake_members):
            result = json.loads(tt.trello_handler(action="list_members", board_id="b1"))
        assert result["count"] == 2
        assert result["members"][0]["username"] == "alice"


# ---------------------------------------------------------------------------
# Action — list_labels
# ---------------------------------------------------------------------------

class TestListLabels:
    def test_success(self, with_creds):
        fake_labels = [
            {"id": "lbl1", "name": "Bug", "color": "red"},
            {"id": "lbl2", "name": "Feature", "color": "green"},
        ]
        with _mock_urlopen(fake_labels):
            result = json.loads(tt.trello_handler(action="list_labels", board_id="b1"))
        assert result["count"] == 2
        assert result["labels"][1]["name"] == "Feature"


# ---------------------------------------------------------------------------
# Action — search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_success(self, with_creds):
        fake_results = {
            "boards": [{"id": "b1", "name": "Board One", "url": "https://trello.com/b1", "closed": False}],
            "cards": [{"id": "c1", "name": "Card One", "url": "https://trello.com/c/c1", "idBoard": "b1", "idList": "l1"}],
        }
        with _mock_urlopen(fake_results):
            result = json.loads(tt.trello_handler(action="search", query="trello"))
        assert result["total_boards"] == 1
        assert result["total_cards"] == 1
        assert result["query"] == "trello"


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

class TestHTTPErrors:
    def test_401_unauthorized(self, with_creds):
        with _mock_http_error(401, "invalid token"):
            result = json.loads(tt.trello_handler(action="list_boards"))
        assert "error" in result
        assert "401" in result["error"]
        assert "TRELLO_API_KEY" in result["error"] or "unauthorized" in result["error"].lower()

    def test_403_forbidden(self, with_creds):
        with _mock_http_error(403, "not authorized"):
            result = json.loads(tt.trello_handler(action="get_board", board_id="b1"))
        assert "error" in result
        assert "403" in result["error"]

    def test_404_not_found(self, with_creds):
        with _mock_http_error(404, "board not found"):
            result = json.loads(tt.trello_handler(action="get_board", board_id="missing"))
        assert "error" in result
        assert "404" in result["error"]

    def test_generic_http_error(self, with_creds):
        with _mock_http_error(500, "internal server error"):
            result = json.loads(tt.trello_handler(action="list_boards"))
        assert "error" in result
        assert "500" in result["error"]


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_has_required_fields(self):
        schema = tt._SCHEMA
        assert schema["name"] == "trello"
        assert "description" in schema
        assert "parameters" in schema
        assert "action" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["action"]

    def test_all_actions_in_enum(self):
        enum_actions = tt._SCHEMA["parameters"]["properties"]["action"]["enum"]
        for action in tt._ACTIONS:
            assert action in enum_actions, f"Action '{action}' missing from schema enum"
