"""``approval`` request params and answers for the ``access`` (folder) and ``network`` (site) kinds."""

import pytest
from pydantic import ValidationError

from tui_gateway.contracts import registry as contracts
from tui_gateway.contracts.server_requests import ApprovalRequestParams, ApprovalResult

_IDS = {"session_id": "s1", "request_id": "r1"}


def _problem(params: dict) -> str | None:
    _, problem = contracts.validate_params(contracts.SERVER_REQUESTS["approval"], {**_IDS, **params})
    return problem


def test_command_approval_needs_no_kind():
    assert _problem({"command": "rm -rf build", "description": "dangerous command"}) is None
    assert ApprovalRequestParams.model_validate({**_IDS, "command": "ls"}).kind is None


def test_folder_and_site_cards_validate():
    assert _problem({"kind": "access", "path": "C:\\docs", "access": "read_write",
                     "risk": "The agent can change and delete files in this folder."}) is None
    assert _problem({"kind": "network", "host": "api.keenable.ai", "label": "Web search (api.keenable.ai)",
                     "choices": ["once", "always", "deny"]}) is None

    folder = ApprovalRequestParams.model_validate({**_IDS, "kind": "access", "path": "/srv", "access": "read"})
    assert (folder.kind, folder.path, folder.access) == ("access", "/srv", "read")
    site = ApprovalRequestParams.model_validate({**_IDS, "kind": "network", "host": "example.com", "label": "Example"})
    assert (site.kind, site.host, site.label) == ("network", "example.com", "Example")


@pytest.mark.parametrize("params", [{"kind": "printer"}, {"kind": "access", "path": "/srv", "access": "rw"}])
def test_values_outside_the_enums_are_refused(params):
    with pytest.raises(ValidationError):
        ApprovalRequestParams.model_validate({**_IDS, **params})


def test_a_deny_reason_rides_the_answer_and_approval_respond():
    assert ApprovalResult.model_validate({"choice": "deny", "reason": "Not this site"}).reason == "Not this site"
    assert ApprovalResult.model_validate({"choice": "once"}).reason is None

    respond = contracts.METHODS["approval.respond"]
    _, problem = contracts.validate_params(respond, {**_IDS, "choice": "deny", "reason": "Not this site"})
    assert problem is None
    _, problem = contracts.validate_params(respond, {**_IDS, "choice": "deny", "why": "x"})
    assert problem is not None and "why" in problem


@pytest.mark.parametrize("params", [{"kind": "access"}, {"kind": "access", "host": "example.com"},
                                    {"kind": "access", "path": " "},
                                    {"kind": "network"}, {"kind": "network", "path": "/srv"}])
def test_a_folder_or_site_card_without_its_target_is_refused(params):
    """An ``access`` card names a ``path`` and a ``network`` card names a ``host``; a request without
    it would offer Allow for something the user cannot read."""
    with pytest.raises(ValidationError):
        ApprovalRequestParams.model_validate({**_IDS, **params})
