"""Explicit GitHub authority checks for repository-object mutations.

GitHub exposes several independently governed object classes.  A token that
can write an issue comment is not proof that it can create a ref or write
repository contents.  This module keeps that distinction in a small,
dependency-free model that can be used before a mutation and after a 403.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping


class GitHubOperation(str, Enum):
    """Object classes used by the automation paths."""

    REPOSITORY_METADATA_READ = "repository.metadata.read"
    ISSUES_COMMENTS_WRITE = "issues.comments.write"
    PULL_REQUESTS_REVIEWS_WRITE = "pull_requests.reviews.write"
    CONTENTS_WRITE = "contents.write"
    GIT_DATA_REFS_WRITE = "git_data.refs.write"
    PULL_REQUEST_CREATE = "pull_request.create"
    ACTIONS_CONTROL = "actions.control"
    WORKFLOWS_WRITE = "workflows.write"
    CHECKS_STATUSES_READ = "checks.statuses.read"

    @property
    def required_permission(self) -> tuple[str, str] | None:
        """Return the GitHub App permission required by this operation."""
        return {
            GitHubOperation.ISSUES_COMMENTS_WRITE: ("issues", "write"),
            GitHubOperation.PULL_REQUESTS_REVIEWS_WRITE: ("pull_requests", "write"),
            GitHubOperation.CONTENTS_WRITE: ("contents", "write"),
            # Git refs are governed by the Contents permission in an App token.
            GitHubOperation.GIT_DATA_REFS_WRITE: ("contents", "write"),
            GitHubOperation.PULL_REQUEST_CREATE: ("pull_requests", "write"),
            GitHubOperation.ACTIONS_CONTROL: ("actions", "write"),
            GitHubOperation.WORKFLOWS_WRITE: ("workflows", "write"),
            GitHubOperation.CHECKS_STATUSES_READ: ("checks", "read"),
        }.get(self)


class GitHubAuthorizationReason(str, Enum):
    """Stable reasons shown to callers and used for recovery guidance."""

    APP_NOT_INSTALLED = "app_not_installed"
    REPOSITORY_NOT_SELECTED = "repository_not_selected"
    PERMISSION_NOT_GRANTED = "permission_not_granted"
    BRANCH_PROTECTION_DENIED = "branch_protection_denied"
    ORGANIZATION_POLICY_DENIED = "organization_policy_denied"
    RESOURCE_NOT_ACCESSIBLE_UNKNOWN = "resource_not_accessible_unknown"


@dataclass(frozen=True)
class GitHubAuthorityContext:
    """Evidence gathered for one repository and one authentication principal."""

    repository: str
    account: str = ""
    organization: str = ""
    authenticated: bool = False
    app_principal: bool = False
    installation_id: str | None = None
    installation_exists: bool | None = None
    repository_selected: bool | None = None
    declared_permissions: Mapping[str, str] = field(default_factory=dict)
    installation_permissions: Mapping[str, str] = field(default_factory=dict)
    repository_permissions: Mapping[str, Any] = field(default_factory=dict)
    operation_grants: Mapping[str, bool] = field(default_factory=dict)
    repository_readable: bool | None = None
    branch_protected: bool = False
    organization_policy_denied: bool = False


def _permission_value(permissions: Mapping[str, Any], name: str) -> str | None:
    value = permissions.get(name)
    if isinstance(value, Mapping):
        value = value.get("level") or value.get("permission")
    if isinstance(value, bool):
        # Repository metadata exposes permissions such as push as booleans;
        # App manifests and installation tokens expose read/write strings.
        return "write" if value else "read"
    if value is None:
        return None
    return str(value).lower()


def _permission_allows(actual: str | None, required: str) -> bool:
    if actual is None:
        return False
    if required == "read":
        return actual in {"read", "write", "admin"}
    return actual in {"write", "admin"}


@dataclass(frozen=True)
class GitHubCapability:
    """Capability result for one exact GitHub operation class."""

    repository: str
    operation: GitHubOperation
    account: str = ""
    organization: str = ""
    app_principal: bool = False
    installation_id: str | None = None
    repository_selected: bool | None = None
    app_declared_permission: str | None = None
    installation_granted_permission: str | None = None
    effective: bool = False
    reason: GitHubAuthorizationReason | None = None
    status_code: int | None = None
    message: str = ""

    @property
    def required_permission(self) -> tuple[str, str] | None:
        return self.operation.required_permission

    @property
    def recovery_action(self) -> str:
        if self.effective:
            return "No recovery action is required."
        if self.reason is GitHubAuthorizationReason.APP_NOT_INSTALLED:
            return (
                "Install the GitHub App in the target account or organization, "
                "then retry the operation."
            )
        if self.reason is GitHubAuthorizationReason.REPOSITORY_NOT_SELECTED:
            return (
                "Select this repository in the GitHub App installation, "
                "then retry the operation."
            )
        if self.reason is GitHubAuthorizationReason.PERMISSION_NOT_GRANTED:
            permission = self.required_permission
            requested = f"{permission[0]}:{permission[1]}" if permission else "the required permission"
            return f"Grant the GitHub App {requested} permission for this repository, then retry."
        if self.reason is GitHubAuthorizationReason.BRANCH_PROTECTION_DENIED:
            return (
                "Use an allowed branch or complete the repository's required "
                "reviews and status checks before retrying."
            )
        if self.reason is GitHubAuthorizationReason.ORGANIZATION_POLICY_DENIED:
            return (
                "Ask an organization administrator to allow this operation "
                "for the principal and repository."
            )
        return (
            "Inspect the target repository, installation selection, and exact "
            "operation permission; do not retry blindly."
        )

    def with_failure(self, status_code: int, message: str) -> "GitHubCapability":
        """Attach a server response and classify it without losing evidence."""
        known_reason = self.reason in {
            GitHubAuthorizationReason.APP_NOT_INSTALLED,
            GitHubAuthorizationReason.REPOSITORY_NOT_SELECTED,
            GitHubAuthorizationReason.PERMISSION_NOT_GRANTED,
            GitHubAuthorizationReason.BRANCH_PROTECTION_DENIED,
            GitHubAuthorizationReason.ORGANIZATION_POLICY_DENIED,
        }
        reason = self.reason if known_reason else classify_github_403(
            status_code,
            message,
            app_principal=self.app_principal,
            repository_selected=self.repository_selected,
        )
        return replace(
            self,
            effective=False,
            reason=reason,
            status_code=status_code,
            message=message[:500],
        )

    def to_dict(self) -> dict[str, Any]:
        permission = self.required_permission
        return {
            "repository": self.repository,
            "account": self.account,
            "organization": self.organization,
            "app_principal": self.app_principal,
            "installation_id": self.installation_id,
            "repository_selected": self.repository_selected,
            "operation_class": self.operation.value,
            "app_declared_permission": self.app_declared_permission,
            "installation_granted_permission": self.installation_granted_permission,
            "required_permission": (
                {"name": permission[0], "level": permission[1]} if permission else None
            ),
            "effective": self.effective,
            "reason": self.reason.value if self.reason else None,
            "status_code": self.status_code,
            "message": self.message,
            "recovery_action": self.recovery_action,
        }


@dataclass(frozen=True)
class GitHubCapabilityReport:
    """First-class report for all requested operation classes."""

    repository: str
    capabilities: tuple[GitHubCapability, ...]

    def for_operation(self, operation: GitHubOperation) -> GitHubCapability:
        for capability in self.capabilities:
            if capability.operation is operation:
                return capability
        raise KeyError(operation.value)

    def require(
        self,
        operations: Iterable[GitHubOperation] | None = None,
    ) -> "GitHubCapabilityReport":
        requested = set(operations or (capability.operation for capability in self.capabilities))
        failures = [
            capability for capability in self.capabilities
            if capability.operation in requested and not capability.effective
        ]
        if failures:
            raise GitHubCapabilityError(failures[0])
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
        }


def _unknown_capability(context: GitHubAuthorityContext, operation: GitHubOperation) -> GitHubCapability:
    permission = operation.required_permission
    return GitHubCapability(
        repository=context.repository,
        operation=operation,
        account=context.account,
        organization=context.organization,
        app_principal=context.app_principal,
        installation_id=context.installation_id,
        repository_selected=context.repository_selected,
        app_declared_permission=(
            _permission_value(context.declared_permissions, permission[0])
            if permission else None
        ),
        installation_granted_permission=(
            _permission_value(context.installation_permissions, permission[0])
            if permission else None
        ),
        reason=GitHubAuthorizationReason.RESOURCE_NOT_ACCESSIBLE_UNKNOWN,
    )


def build_capability_report(
    context: GitHubAuthorityContext,
    operations: Iterable[GitHubOperation] | None = None,
) -> GitHubCapabilityReport:
    """Build a fail-closed report from known installation/repository evidence.

    App tokens are evaluated against declared and installation-granted
    permissions. PAT/gh tokens use repository permissions for repository
    contents and require explicit operation grants for separately governed
    Actions/workflow controls. No successful metadata operation is used as a
    proxy for a contents or ref write.
    """
    requested = tuple(operations or GitHubOperation)
    results: list[GitHubCapability] = []

    for operation in requested:
        base = _unknown_capability(context, operation)
        permission = operation.required_permission
        declared = base.app_declared_permission
        granted = base.installation_granted_permission

        if context.app_principal:
            if not context.installation_id or context.installation_exists is False:
                results.append(replace(
                    base,
                    reason=GitHubAuthorizationReason.APP_NOT_INSTALLED,
                ))
                continue
            if context.repository_selected is False:
                results.append(replace(
                    base,
                    reason=GitHubAuthorizationReason.REPOSITORY_NOT_SELECTED,
                ))
                continue
            if context.installation_exists is not True or context.repository_selected is not True:
                results.append(base)
                continue

        if not context.authenticated:
            results.append(base)
            continue

        explicit = context.operation_grants.get(operation.value)
        if explicit is False:
            results.append(replace(
                base,
                reason=GitHubAuthorizationReason.PERMISSION_NOT_GRANTED,
            ))
            continue
        if explicit is True and context.authenticated:
            results.append(replace(base, effective=True, reason=None))
            continue

        if context.app_principal:
            if context.organization_policy_denied:
                results.append(replace(
                    base,
                    reason=GitHubAuthorizationReason.ORGANIZATION_POLICY_DENIED,
                ))
                continue
            if context.branch_protected and operation in {
                GitHubOperation.CONTENTS_WRITE,
                GitHubOperation.GIT_DATA_REFS_WRITE,
            }:
                results.append(replace(
                    base,
                    reason=GitHubAuthorizationReason.BRANCH_PROTECTION_DENIED,
                ))
                continue
            if permission and (
                not _permission_allows(declared, permission[1])
                or not _permission_allows(granted, permission[1])
            ):
                results.append(replace(
                    base,
                    reason=GitHubAuthorizationReason.PERMISSION_NOT_GRANTED,
                ))
                continue
            if operation is GitHubOperation.REPOSITORY_METADATA_READ:
                effective = context.repository_readable is True
            else:
                effective = context.repository_readable is not False
            results.append(replace(
                base,
                effective=effective,
                reason=None if effective else GitHubAuthorizationReason.RESOURCE_NOT_ACCESSIBLE_UNKNOWN,
            ))
            continue

        if context.organization_policy_denied:
            results.append(replace(
                base,
                reason=GitHubAuthorizationReason.ORGANIZATION_POLICY_DENIED,
            ))
            continue
        if context.branch_protected and operation in {
            GitHubOperation.CONTENTS_WRITE,
            GitHubOperation.GIT_DATA_REFS_WRITE,
        }:
            results.append(replace(
                base,
                reason=GitHubAuthorizationReason.BRANCH_PROTECTION_DENIED,
            ))
            continue

        readable = context.repository_readable is True
        if operation is GitHubOperation.REPOSITORY_METADATA_READ:
            effective = readable
        elif operation is GitHubOperation.PULL_REQUEST_CREATE:
            # A contributor can create a PR against a readable base repository
            # while writing code only to a separately authorized fork.
            effective = readable
        elif operation in {
            GitHubOperation.CONTENTS_WRITE,
            GitHubOperation.GIT_DATA_REFS_WRITE,
        }:
            effective = _permission_allows(
                _permission_value(context.repository_permissions, "push"),
                "write",
            )
        elif explicit is True:
            effective = True
        else:
            effective = False

        results.append(replace(
            base,
            effective=effective,
            reason=None if effective else GitHubAuthorizationReason.RESOURCE_NOT_ACCESSIBLE_UNKNOWN,
        ))

    return GitHubCapabilityReport(
        repository=context.repository,
        capabilities=tuple(results),
    )


def classify_github_403(
    status_code: int,
    message: str,
    *,
    app_principal: bool = False,
    repository_selected: bool | None = None,
    branch_protected: bool = False,
    organization_policy_denied: bool = False,
) -> GitHubAuthorizationReason:
    """Classify a forbidden response only when the evidence supports it."""
    if status_code != 403:
        return GitHubAuthorizationReason.RESOURCE_NOT_ACCESSIBLE_UNKNOWN

    lowered = (message or "").lower()
    if organization_policy_denied or any(marker in lowered for marker in (
        "organization policy",
        "enterprise policy",
        "saml",
        "ip allow list",
        "ip allowlist",
    )):
        return GitHubAuthorizationReason.ORGANIZATION_POLICY_DENIED
    if branch_protected or any(marker in lowered for marker in (
        "protected branch",
        "required status check",
        "review required",
        "branch protection",
    )):
        return GitHubAuthorizationReason.BRANCH_PROTECTION_DENIED
    if repository_selected is False or any(marker in lowered for marker in (
        "repository not selected",
        "not selected for this installation",
    )):
        return GitHubAuthorizationReason.REPOSITORY_NOT_SELECTED
    if app_principal and any(marker in lowered for marker in (
        "app is not installed",
        "installation not found",
        "not installed",
    )):
        return GitHubAuthorizationReason.APP_NOT_INSTALLED
    if any(marker in lowered for marker in (
        "insufficient permission",
        "insufficient scope",
        "permission denied",
        "missing permission",
    )):
        return GitHubAuthorizationReason.PERMISSION_NOT_GRANTED
    return GitHubAuthorizationReason.RESOURCE_NOT_ACCESSIBLE_UNKNOWN


class GitHubCapabilityError(RuntimeError):
    """Raised when a mutation has no effective authority preflight."""

    def __init__(self, capability: GitHubCapability):
        self.capability = capability
        super().__init__(
            f"GitHub authority preflight failed for {capability.operation.value} "
            f"on {capability.repository}: "
            f"{capability.reason.value if capability.reason else 'not_effective'}. "
            f"{capability.recovery_action}"
        )


@dataclass(frozen=True)
class CodePublicationReceipt:
    """Evidence that a code publication produced the expected source object."""

    repository: str
    branch: str
    commit_sha: str
    source_diff_verified: bool
    pull_request_url: str | None = None

    @property
    def is_complete(self) -> bool:
        return bool(
            self.repository.strip()
            and self.branch.strip()
            and self.commit_sha.strip()
            and self.source_diff_verified
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "source_diff_verified": self.source_diff_verified,
            "pull_request_url": self.pull_request_url,
            "complete": self.is_complete,
        }


class CodePublicationReceiptError(RuntimeError):
    """Raised when a publisher cannot prove the resulting source change."""


def require_code_publication_receipt(
    receipt: CodePublicationReceipt,
) -> CodePublicationReceipt:
    if not receipt.is_complete:
        raise CodePublicationReceiptError(
            "Code publication did not produce a complete receipt: "
            "repository, branch, commit SHA, and verified source diff are required."
        )
    return receipt


def is_code_publication_complete(receipt: CodePublicationReceipt | None) -> bool:
    return bool(receipt and receipt.is_complete)
