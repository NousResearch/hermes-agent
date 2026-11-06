"""Credential-free account limits of one authorized session, never an aggregate of pool logins."""
from typing import Literal

from pydantic import Field, model_validator

from .base import Result
from .common import ProfileParams
from .registry import canonical_method


class AccountUsageRowArgs(Result):
    value: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    count: int | None = Field(default=None, ge=0, le=10**15)
    used: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    limit: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    total: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    daily: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    weekly: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    monthly: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)


class AccountUsageRowView(Result):
    key: Literal["credits_balance", "banked_resets", "credits_unlimited", "extra_usage", "api_key_usage"]
    args: AccountUsageRowArgs


class AccountUsageWindowView(Result):
    label: str
    label_key: Literal["session", "weekly", "current_session", "current_week", "opus_week", "sonnet_week", "api_key_quota"]
    scope: Literal["account", "model"]
    used_percent: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    remaining_percent: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    reset_at: str | None = None
    limit: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    limit_remaining: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    reset_interval: Literal["daily", "weekly", "monthly"] | None = None


class AccountUsageSnapshotView(Result):
    """Only normalized numbers and fixed labels cross the wire; raw, account IDs and URLs stay private."""

    provider: Literal["openrouter", "openai-codex", "anthropic"]
    fetched_at: str
    plan: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9 _-]{0,63}$")
    windows: list[AccountUsageWindowView]
    rows: list[AccountUsageRowView]
    credits_balance: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    usage_window: bool = False


class SessionAccountUsageParams(ProfileParams):
    session_id: str = Field(min_length=1)


class SessionAccountUsageResult(Result):
    status: Literal["ok", "unsupported", "unavailable"]
    provider: Literal["openrouter", "openai-codex", "anthropic"] | None = None
    reason: Literal["provider_unsupported", "route_unsupported", "oauth_required", "credentials_unavailable",
                    "credentials_changed", "fetch_failed", "usage_unavailable"] | None = None
    snapshot: AccountUsageSnapshotView | None = None

    @model_validator(mode="after")
    def _status_matches_snapshot(self):
        if self.status == "ok":
            if self.snapshot is None or self.reason is not None or self.provider != self.snapshot.provider:
                raise ValueError("ok account usage requires its provider snapshot and no failure reason")
        elif self.snapshot is not None or self.reason is None:
            raise ValueError("unsupported or unavailable account usage requires a reason and no snapshot")
        return self


canonical_method("session.account_usage", params=SessionAccountUsageParams, result=SessionAccountUsageResult,
                 doc="Read balance and quota for the exact retained session credential, without inference or auth rotation.")
