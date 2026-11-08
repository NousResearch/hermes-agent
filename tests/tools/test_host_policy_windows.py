"""Native Windows fails closed until account-token/DACL support is qualified."""

import pytest

from tools import approval, host_policy
from tools.host_policy import HostPolicyError
from tools.worker_admission import reserve_workers

pytestmark = pytest.mark.platforms("windows")


def test_absent_authority_remains_unmanaged(tmp_path, monkeypatch):
    monkeypatch.setattr(host_policy, "policy_root", lambda: tmp_path / "absent")
    assert host_policy.load_host_policy() is None
    assert reserve_workers(1) == []


def test_installed_authority_is_refused_without_acl_qualification(tmp_path, monkeypatch):
    root = tmp_path / "authority"
    root.mkdir()
    monkeypatch.setattr(host_policy, "policy_root", lambda: root)
    with pytest.raises(HostPolicyError, match="Windows ACL"):
        host_policy.load_host_policy()
    with pytest.raises(HostPolicyError):
        reserve_workers(1)
    result = approval.check_all_command_guards("dir", "local")
    assert result["approved"] is False
    assert result["outcome"] == "policy_error"
