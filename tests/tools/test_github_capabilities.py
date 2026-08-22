"""Regression tests for GitHub object-class authority and publication receipts."""

import pytest

from tools.github_capabilities import (
    CodePublicationReceipt,
    GitHubAuthorityContext,
    GitHubAuthorizationReason,
    GitHubCapabilityError,
    GitHubOperation,
    build_capability_report,
    classify_github_403,
    require_code_publication_receipt,
)


def test_github_auth_report_uses_installation_selection_and_grants(monkeypatch):
    import time
    import tools.skills_hub as skills_hub

    auth = skills_hub.GitHubAuth()
    auth._cached_token = "test-token"
    auth._cached_method = "github-app"
    auth._app_token_expiry = time.time() + 3600
    auth._app_principal_configured = True
    auth._app_installation_id = "123"
    auth._app_installation_exists = True
    auth._app_repository_selection = "selected"
    auth._app_selected_repositories = {"owner/repository"}
    auth._app_declared_permissions = {"contents": "write"}
    auth._app_granted_permissions = {"contents": "write"}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "owner": {"login": "owner", "type": "User"},
                "permissions": {"pull": True, "push": True},
            }

    monkeypatch.setattr(skills_hub.httpx, "get", lambda *_args, **_kwargs: Response())
    capability = auth.capability_report(
        "owner/repository",
        [GitHubOperation.CONTENTS_WRITE],
    ).for_operation(GitHubOperation.CONTENTS_WRITE)

    assert capability.effective
    assert capability.app_principal
    assert capability.repository_selected is True
    assert capability.installation_id == "123"


def _app_context(**overrides):
    values = {
        "repository": "owner/repository",
        "authenticated": True,
        "app_principal": True,
        "installation_id": "123",
        "installation_exists": True,
        "repository_selected": True,
        "repository_readable": True,
        "declared_permissions": {
            "contents": "write",
            "pull_requests": "write",
            "issues": "write",
            "actions": "write",
        },
        "installation_permissions": {
            "contents": "write",
            "pull_requests": "write",
            "issues": "write",
            "actions": "write",
        },
    }
    values.update(overrides)
    return GitHubAuthorityContext(**values)


def test_app_without_installation_is_reported_before_mutation():
    report = build_capability_report(
        _app_context(installation_id=None, installation_exists=False),
        [GitHubOperation.CONTENTS_WRITE],
    )

    capability = report.for_operation(GitHubOperation.CONTENTS_WRITE)
    assert not capability.effective
    assert capability.reason is GitHubAuthorizationReason.APP_NOT_INSTALLED
    with pytest.raises(GitHubCapabilityError, match="Install the GitHub App"):
        report.require()


def test_configured_app_without_installation_is_typed_even_without_token():
    report = build_capability_report(
        GitHubAuthorityContext(
            repository="owner/repository",
            app_principal=True,
            installation_exists=False,
        ),
        [GitHubOperation.CONTENTS_WRITE],
    )

    assert report.for_operation(
        GitHubOperation.CONTENTS_WRITE
    ).reason is GitHubAuthorizationReason.APP_NOT_INSTALLED


def test_selected_repository_is_required_even_when_app_permission_is_granted():
    report = build_capability_report(
        _app_context(repository_selected=False),
        [GitHubOperation.CONTENTS_WRITE],
    )
    capability = report.for_operation(GitHubOperation.CONTENTS_WRITE)

    assert capability.reason is GitHubAuthorizationReason.REPOSITORY_NOT_SELECTED
    assert "Select this repository" in capability.recovery_action


def test_declared_write_without_granted_write_is_not_effective():
    report = build_capability_report(
        _app_context(installation_permissions={"contents": "read"}),
        [GitHubOperation.CONTENTS_WRITE],
    )
    capability = report.for_operation(GitHubOperation.CONTENTS_WRITE)

    assert not capability.effective
    assert capability.reason is GitHubAuthorizationReason.PERMISSION_NOT_GRANTED


def test_operation_classes_do_not_inherit_authority_from_one_another():
    report = build_capability_report(
        _app_context(
            installation_permissions={
                "contents": "write",
                "pull_requests": "write",
                "issues": "write",
                "actions": "read",
            }
        ),
        [
            GitHubOperation.ISSUES_COMMENTS_WRITE,
            GitHubOperation.CONTENTS_WRITE,
            GitHubOperation.ACTIONS_CONTROL,
        ],
    )

    assert report.for_operation(GitHubOperation.ISSUES_COMMENTS_WRITE).effective
    assert report.for_operation(GitHubOperation.CONTENTS_WRITE).effective
    actions = report.for_operation(GitHubOperation.ACTIONS_CONTROL)
    assert not actions.effective
    assert actions.reason is GitHubAuthorizationReason.PERMISSION_NOT_GRANTED


def test_pat_can_create_cross_repo_pr_without_upstream_code_write():
    context = GitHubAuthorityContext(
        repository="NousResearch/hermes-agent",
        authenticated=True,
        repository_readable=True,
        repository_permissions={"push": False},
    )
    report = build_capability_report(
        context,
        [GitHubOperation.PULL_REQUEST_CREATE, GitHubOperation.CONTENTS_WRITE],
    )

    assert report.for_operation(GitHubOperation.PULL_REQUEST_CREATE).effective
    assert not report.for_operation(GitHubOperation.CONTENTS_WRITE).effective


def test_branch_protection_and_organization_policy_are_distinct():
    branch = build_capability_report(
        _app_context(branch_protected=True),
        [GitHubOperation.CONTENTS_WRITE],
    ).for_operation(GitHubOperation.CONTENTS_WRITE)
    organization = build_capability_report(
        _app_context(organization_policy_denied=True),
        [GitHubOperation.PULL_REQUEST_CREATE],
    ).for_operation(GitHubOperation.PULL_REQUEST_CREATE)

    assert branch.reason is GitHubAuthorizationReason.BRANCH_PROTECTION_DENIED
    assert organization.reason is GitHubAuthorizationReason.ORGANIZATION_POLICY_DENIED


def test_generic_resource_not_accessible_403_stays_unknown():
    reason = classify_github_403(
        403,
        "Resource not accessible by integration",
        app_principal=True,
    )

    assert reason is GitHubAuthorizationReason.RESOURCE_NOT_ACCESSIBLE_UNKNOWN


def test_known_preflight_reason_survives_generic_403_text():
    capability = build_capability_report(
        _app_context(installation_exists=False),
        [GitHubOperation.CONTENTS_WRITE],
    ).for_operation(GitHubOperation.CONTENTS_WRITE)

    failed = capability.with_failure(403, "Resource not accessible by integration")
    assert failed.reason is GitHubAuthorizationReason.APP_NOT_INSTALLED


def test_typed_403_evidence_is_preserved():
    assert classify_github_403(
        403, "protected branch hook declined", branch_protected=False
    ) is GitHubAuthorizationReason.BRANCH_PROTECTION_DENIED
    assert classify_github_403(
        403, "installation not found", app_principal=True
    ) is GitHubAuthorizationReason.APP_NOT_INSTALLED
    assert classify_github_403(
        403, "insufficient permission for contents"
    ) is GitHubAuthorizationReason.PERMISSION_NOT_GRANTED


def test_capability_report_exposes_authority_evidence():
    capability = build_capability_report(
        _app_context(),
        [GitHubOperation.CONTENTS_WRITE],
    ).for_operation(GitHubOperation.CONTENTS_WRITE)
    payload = capability.to_dict()

    assert payload["operation_class"] == "contents.write"
    assert payload["installation_id"] == "123"
    assert payload["app_declared_permission"] == "write"
    assert payload["installation_granted_permission"] == "write"
    assert payload["effective"] is True


def test_code_publication_receipt_requires_commit_and_verified_diff():
    incomplete = CodePublicationReceipt(
        repository="owner/repository",
        branch="change",
        commit_sha="",
        source_diff_verified=False,
    )
    with pytest.raises(RuntimeError, match="complete receipt"):
        require_code_publication_receipt(incomplete)

    complete = require_code_publication_receipt(CodePublicationReceipt(
        repository="owner/repository",
        branch="change",
        commit_sha="abc123",
        source_diff_verified=True,
        pull_request_url="https://github.com/owner/repository/pull/1",
    ))
    assert complete.is_complete
