"""
Nacos 3.2 Skills Registry adapter for hermes-agent's skills hub.

Identifier formats accepted:

    nacos://<namespace>/<group>/<name>[@<version>]
    <name>                                          (defaults: public/hermes-skills/latest)

The adapter delegates network/protocol work to ``NacosCliClient`` (which
wraps the Node ``nacos-cli`` binary).  ZIP payloads returned by nacos-cli
are unpacked in-memory into a ``SkillBundle`` so the rest of the hub
(quarantine scanning, install_from_quarantine, HubLockFile) works unchanged.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Union

from tools.nacos_cli_client import (
    NacosCliClient,
    NacosCliError,
    NacosNotFound,
    NacosSkillEntry,
)
from tools.skills_hub import SkillBundle, SkillMeta, SkillSource

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "public"
DEFAULT_GROUP = "hermes-skills"


@dataclass(frozen=True)
class NacosIdentifier:
    namespace: str
    group: str
    name: str
    version: Optional[str]

    def canonical(self) -> str:
        tail = f"@{self.version}" if self.version else ""
        return f"nacos://{self.namespace}/{self.group}/{self.name}{tail}"


_IDENT_RE = re.compile(
    r"^nacos://(?P<ns>[^/]+)/(?P<group>[^/]+)/(?P<name>[^@/]+)"
    r"(?:@(?P<ver>[^/]+))?$"
)


def parse_nacos_identifier(value: str) -> NacosIdentifier:
    """Parse a nacos identifier; accept bare names (defaults applied)."""
    if not value:
        raise ValueError("empty nacos identifier")
    if not value.startswith("nacos://"):
        return NacosIdentifier(DEFAULT_NAMESPACE, DEFAULT_GROUP, value, None)
    m = _IDENT_RE.match(value)
    if not m:
        raise ValueError(f"invalid nacos identifier: {value!r}")
    return NacosIdentifier(
        namespace=m.group("ns"),
        group=m.group("group"),
        name=m.group("name"),
        version=m.group("ver"),
    )


class NacosSkillSource(SkillSource):
    """``SkillSource`` adapter for Nacos 3.2 Skills Registry."""

    def __init__(
        self,
        client: Optional[NacosCliClient] = None,
        *,
        default_namespace: Optional[str] = None,
        default_group: Optional[str] = None,
        trusted_namespaces: Optional[List[str]] = None,
    ):
        self.client = client or NacosCliClient()
        self.default_namespace = default_namespace or os.environ.get(
            "NACOS_NAMESPACE", DEFAULT_NAMESPACE
        )
        self.default_group = default_group or DEFAULT_GROUP
        self.trusted_namespaces = set(trusted_namespaces or [])

    def source_id(self) -> str:
        return "nacos"

    # ------------------------------------------------------------------ search

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        try:
            entries = self.client.list_skills(
                namespace=self.default_namespace,
                group=self.default_group,
                query=query or None,
                limit=limit,
            )
        except NacosCliError as e:
            logger.warning("nacos search failed: %s", e)
            return []
        return [self._entry_to_meta(e) for e in entries]

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        try:
            ident = parse_nacos_identifier(identifier)
        except ValueError:
            return None
        try:
            entries = self.client.list_skills(
                namespace=ident.namespace,
                group=ident.group,
                query=ident.name,
                limit=20,
            )
        except NacosNotFound:
            return None
        except NacosCliError as e:
            logger.warning("nacos inspect failed: %s", e)
            return None
        for entry in entries:
            if entry.name == ident.name:
                return self._entry_to_meta(entry, override_ident=ident)
        return None

    def trust_level_for(self, identifier: str) -> str:
        try:
            ident = parse_nacos_identifier(identifier)
        except ValueError:
            return "community"
        if ident.namespace in self.trusted_namespaces:
            return "trusted"
        return "community"

    # ------------------------------------------------------------------ fetch

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        try:
            ident = parse_nacos_identifier(identifier)
        except ValueError:
            return None

        meta = self.inspect(identifier)
        if meta is None:
            return None

        with tempfile.TemporaryDirectory() as td:
            try:
                zip_path, checksum = self.client.get_skill(
                    ident.name,
                    namespace=ident.namespace,
                    group=ident.group,
                    version=ident.version,
                    output_dir=Path(td),
                )
            except NacosNotFound:
                return None
            except NacosCliError as e:
                logger.warning("nacos fetch failed: %s", e)
                return None

            try:
                files = self._extract_zip_safe(zip_path)
            except ValueError as e:
                # Unsafe ZIP contents (path traversal, absolute path).
                # Surface as NacosCliError so the CLI layer prints a clean
                # error instead of a traceback.
                raise NacosCliError(f"unsafe nacos zip for {ident.name}: {e}") from e

        return SkillBundle(
            name=ident.name,
            files=files,
            source="nacos",
            identifier=ident.canonical(),
            trust_level=self.trust_level_for(identifier),
            metadata={
                "namespace": ident.namespace,
                "group": ident.group,
                "version": meta.extra.get("version"),
                "checksum": checksum or meta.extra.get("checksum"),
            },
        )

    # ------------------------------------------------------------------ helpers

    def _entry_to_meta(
        self,
        entry: NacosSkillEntry,
        *,
        override_ident: Optional[NacosIdentifier] = None,
    ) -> SkillMeta:
        ident = override_ident or NacosIdentifier(
            entry.namespace, entry.group, entry.name, None
        )
        canonical = NacosIdentifier(
            ident.namespace, ident.group, entry.name, None
        ).canonical()
        trust = "trusted" if ident.namespace in self.trusted_namespaces else "community"
        return SkillMeta(
            name=entry.name,
            description=entry.description,
            source="nacos",
            identifier=canonical,
            trust_level=trust,
            extra={
                "version": entry.version,
                "author": entry.author,
                "updated_at": entry.updated_at,
                "checksum": entry.checksum,
            },
        )

    @staticmethod
    def _extract_zip_safe(zip_path: Path) -> Dict[str, Union[str, bytes]]:
        """Read ZIP entries into a ``{relpath: str|bytes}`` dict; reject traversal."""
        out: Dict[str, Union[str, bytes]] = {}
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if name.startswith("/"):
                    raise ValueError(f"unsafe path in nacos zip: {name!r}")
                parts = PurePosixPath(name).parts
                if any(p == ".." for p in parts):
                    raise ValueError(f"unsafe path in nacos zip: {name!r}")
                data = zf.read(info)
                try:
                    out[name] = data.decode("utf-8")
                except UnicodeDecodeError:
                    out[name] = data
        return out
