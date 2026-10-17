"""Client policy remains mandatory across local bypasses and concurrent sessions."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tools.command_permissions import reset_command_permission_requester, set_command_permission_requester
from tools.terminal_tool import _Rejected, _run_approval_guards


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("choice", [True, False, "error"])
def test_client_policy_applies_even_to_safe_or_force_approved_commands(force, choice):
    requests = []

    def requester(command):
        requests.append(command)
        if choice == "error":
            raise RuntimeError("connection lost")
        return choice

    token = set_command_permission_requester(requester)
    try:
        if choice is True:
            _run_approval_guards("printf safe", "local", {}, force=force)
        else:
            with pytest.raises(_Rejected):
                _run_approval_guards("printf safe", "local", {}, force=force)
        assert requests == ["printf safe"]
    finally:
        reset_command_permission_requester(token)
    # A reused worker with no ACP owner retains the ordinary local behavior.
    _run_approval_guards("printf safe", "local", {}, force=force)
    assert requests == ["printf safe"]


def test_concurrent_session_permissions_are_isolated_and_reset():
    barrier = Barrier(2)

    def worker(allowed):
        requests = []
        token = set_command_permission_requester(lambda command: requests.append(command) or allowed)
        try:
            barrier.wait(timeout=10)
            try:
                _run_approval_guards("printf safe", "local", {}, force=True)
                outcome = True
            except _Rejected:
                outcome = False
        finally:
            reset_command_permission_requester(token)
        _run_approval_guards("printf next", "local", {}, force=True)
        return outcome, requests

    with ThreadPoolExecutor(max_workers=2) as executor:
        allowed, denied = executor.submit(worker, True), executor.submit(worker, False)
        assert allowed.result(timeout=10) == (True, ["printf safe"])
        assert denied.result(timeout=10) == (False, ["printf safe"])
