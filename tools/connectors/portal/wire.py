"""Typed wire shapes for connector metadata from the portal."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _PortalWire(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, strict=True)


class ConnectorTool(_PortalWire):
    slug: str
    name: str
    description: str
    facet: Literal["read", "write", "destructive", "unclassified"] = "unclassified"
    hints: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    no_auth: bool = Field(default=False, alias="noAuth")
    deprecated: bool = False

    @field_validator("facet", mode="before")
    @classmethod
    def _unknown_facet_is_unclassified(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value if value in {"read", "write", "destructive", "unclassified"} else "unclassified"

    @field_validator("categories")
    @classmethod
    def _normalize_categories(cls, value: list[str]) -> list[str]:
        return [category.lower() for category in value]


class ConnectorToolsListing(_PortalWire):
    connector: str
    toolkit_version: str = Field(alias="toolkitVersion")
    etag: str
    tools: list[ConnectorTool]


class ConnectorCatalogRow(_PortalWire):
    slug: str
    name: str
    description: str
    category: str
    logo_url: str | None = Field(default=None, alias="logoUrl")


class ConnectorCatalogResponse(_PortalWire):
    connectors: list[ConnectorCatalogRow]


class PolicyTags(_PortalWire):
    enable: list[str] | None = None
    disable: list[str] | None = None


class UnrestrictedPolicyBody(_PortalWire):
    mode: Literal["unrestricted"]


class DenyAllPolicyBody(_PortalWire):
    mode: Literal["deny-all"]


class PolicyToolRule(_PortalWire):
    disable: list[str] = Field(default_factory=list)


class AllowPolicyBody(_PortalWire):
    mode: Literal["allow"]
    connectors: list[str]
    tools: dict[str, PolicyToolRule] = Field(default_factory=dict)
    tags: PolicyTags | None = None


class DenyPolicyBody(_PortalWire):
    mode: Literal["deny"]
    disabled_connectors: list[str] = Field(alias="disabledConnectors")
    tools: dict[str, PolicyToolRule] = Field(default_factory=dict)
    tags: PolicyTags | None = None


PolicyBody = UnrestrictedPolicyBody | DenyAllPolicyBody | AllowPolicyBody | DenyPolicyBody


class ConnectorPolicyLayer(_PortalWire):
    kind: Literal["org", "role", "member"]
    id: str | None = None
    body: PolicyBody = Field(discriminator="mode")
    revision: str


class ConnectorPolicyResponse(_PortalWire):
    layers: list[ConnectorPolicyLayer]


class ConnectorPolicyWriteResponse(_PortalWire):
    revision: str
