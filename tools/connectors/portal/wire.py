from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _PortalWire(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, strict=True)


class ConnectorTool(_PortalWire):
    slug: str
    name: str
    description: str
    facet: Literal["read", "write", "destructive", "unclassified"]
    hints: list[str]
    categories: list[str]
    no_auth: bool = Field(alias="noAuth")
    deprecated: bool

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


class ConnectorPolicyEffectiveBase(_PortalWire):
    version: Literal[1]
    revision: str
    issued_at_ms: int = Field(alias="issuedAtMs")


class ConnectorPolicyEffectiveUnrestricted(ConnectorPolicyEffectiveBase):
    mode: Literal["unrestricted"]


class ConnectorPolicyEffectiveDenyAll(ConnectorPolicyEffectiveBase):
    mode: Literal["deny-all"]


class ConnectorPolicyEffectiveAllow(ConnectorPolicyEffectiveBase):
    mode: Literal["allow"]
    connectors: list[str]
    tools: dict[str, PolicyToolRule] = Field(default_factory=dict)
    tags: PolicyTags | None = None


class ConnectorPolicyEffectiveDeny(ConnectorPolicyEffectiveBase):
    mode: Literal["deny"]
    disabled_connectors: list[str] = Field(alias="disabledConnectors")
    tools: dict[str, PolicyToolRule] = Field(default_factory=dict)
    tags: PolicyTags | None = None


ConnectorPolicyEffective = (
    ConnectorPolicyEffectiveUnrestricted
    | ConnectorPolicyEffectiveDenyAll
    | ConnectorPolicyEffectiveAllow
    | ConnectorPolicyEffectiveDeny
)


class ConnectorPolicyLayer(_PortalWire):
    kind: Literal["org", "role", "member"]
    id: str | None = None
    body: PolicyBody = Field(discriminator="mode")
    revision: str


class ConnectorPolicyResponse(_PortalWire):
    layers: list[ConnectorPolicyLayer]
    effective: ConnectorPolicyEffective = Field(discriminator="mode")


class ConnectorPolicyWriteResponse(_PortalWire):
    revision: str
    effective: ConnectorPolicyEffective = Field(discriminator="mode")
