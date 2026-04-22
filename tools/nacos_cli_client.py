"""
nacos-cli subprocess client for Nacos 3.2 Skills Registry.

hermes acts as a thin client of the Node-based ``@nacos-group/cli`` binary.
This module wraps subprocess calls with structured error types so CLI
subcommands and ``NacosSkillSource`` can handle failures cleanly.

The Node CLI is authoritative for ZIP packaging, signing, upload, download.
hermes never reads ``~/.nacos-cli.conf`` (credentials stay out of our process).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NacosCliError(Exception):
    """Base class for all nacos-cli wrapper errors."""


class NacosCliNotInstalled(NacosCliError):
    """nacos-cli binary not found on PATH."""


class NacosAuthError(NacosCliError):
    """Nacos returned 401/403."""


class NacosNotFound(NacosCliError):
    """Nacos returned 404 for the requested skill."""


class NacosVersionConflict(NacosCliError):
    """Nacos returned 409 for a push with a stale version."""


class NacosTimeout(NacosCliError):
    """nacos-cli timed out."""


class NacosCliOutputError(NacosCliError):
    """nacos-cli stdout is not valid JSON or is malformed."""


@dataclass(frozen=True)
class NacosSkillEntry:
    """A single skill record returned by ``skill-list`` / ``skill-get``."""

    name: str
    namespace: str
    group: str
    version: str
    description: str
    author: Optional[str]
    updated_at: Optional[str]
    checksum: Optional[str]

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "NacosSkillEntry":
        return cls(
            name=obj["name"],
            namespace=obj.get("namespace", "public"),
            group=obj.get("group", "hermes-skills"),
            version=obj.get("version", "latest"),
            description=obj.get("description", ""),
            author=obj.get("author"),
            updated_at=obj.get("updatedAt"),
            checksum=obj.get("checksum"),
        )


class NacosCliClient:
    """subprocess wrapper around ``nacos-cli`` (Node binary from ``@nacos-group/cli``)."""

    DEFAULT_BIN = "nacos-cli"
    DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        bin_path: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.bin_path = bin_path or self.DEFAULT_BIN
        self.timeout = timeout

    # ------------------------------------------------------------------ diagnostics

    def check_installed(self) -> bool:
        """Return True if nacos-cli is executable."""
        path = Path(self.bin_path)
        if path.is_absolute():
            return path.is_file() and (path.stat().st_mode & 0o111) != 0
        return shutil.which(self.bin_path) is not None

    def version(self) -> str:
        """Return nacos-cli version string."""
        try:
            result = subprocess.run(
                [self.bin_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError as e:
            raise NacosCliNotInstalled(
                f"nacos-cli not found at {self.bin_path!r}. "
                "Install via `npm i -g @nacos-group/cli`."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise NacosTimeout("nacos-cli --version timed out") from e
        if result.returncode != 0:
            raise NacosCliError(
                f"nacos-cli --version failed: {(result.stderr or '').strip()}"
            )
        return result.stdout.strip()

    # ------------------------------------------------------------------ internal

    def _run(
        self,
        args: List[str],
        *,
        expect_json: bool = True,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Run nacos-cli with args; map exit codes / stderr to exceptions."""
        cmd = [self.bin_path, *args]
        logger.debug("nacos-cli: %s", cmd)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, **(extra_env or {})},
            )
        except FileNotFoundError as e:
            raise NacosCliNotInstalled(
                f"nacos-cli not found at {self.bin_path!r}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise NacosTimeout(
                f"nacos-cli timed out after {self.timeout}s: {' '.join(args)}"
            ) from e

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            lowered = stderr.lower()
            if (
                "401" in stderr
                or "403" in stderr
                or "unauthorized" in lowered
                or "forbidden" in lowered
            ):
                raise NacosAuthError(stderr or "nacos auth failed")
            if "404" in stderr or "not found" in lowered:
                raise NacosNotFound(stderr or "nacos resource not found")
            if "409" in stderr or "conflict" in lowered:
                raise NacosVersionConflict(stderr or "nacos version conflict")
            raise NacosCliError(
                f"nacos-cli failed ({result.returncode}): {stderr}"
            )

        if not expect_json:
            return result.stdout

        stdout = result.stdout or ""
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            raise NacosCliOutputError(
                f"nacos-cli stdout is not valid JSON: {stdout[:200]!r}"
            ) from e

    # ------------------------------------------------------------------ skill operations

    def list_skills(
        self,
        *,
        namespace: str = "public",
        group: str = "hermes-skills",
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[NacosSkillEntry]:
        """List skills in a Nacos namespace/group."""
        args = [
            "skill-list",
            "--namespace", namespace,
            "--group", group,
            "--limit", str(limit),
            "--json",
        ]
        if query:
            args.extend(["--query", query])
        data = self._run(args)
        return [NacosSkillEntry.from_json(e) for e in data.get("skills", [])]

    def get_skill(
        self,
        name: str,
        *,
        namespace: str = "public",
        group: str = "hermes-skills",
        version: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Optional[str]]:
        """Download a skill ZIP. Returns ``(zip_path, checksum_or_None)``."""
        out_dir = output_dir or Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "skill-get",
            "--namespace", namespace,
            "--group", group,
            "--name", name,
            "--output", str(out_dir),
            "--json",
        ]
        if version:
            args.extend(["--version", version])
        meta = self._run(args)
        file_path = Path(meta["file"])
        return file_path, meta.get("checksum")

    def upload_skill(
        self,
        zip_path: Path,
        *,
        namespace: str = "public",
        group: str = "hermes-skills",
    ) -> Dict[str, Any]:
        """Upload a skill ZIP. Returns the parsed upload result."""
        if not zip_path.exists():
            raise FileNotFoundError(f"skill zip not found: {zip_path}")
        args = [
            "skill-upload",
            "--namespace", namespace,
            "--group", group,
            "--file", str(zip_path),
            "--json",
        ]
        return self._run(args)

    def sync_namespace(
        self,
        *,
        namespace: str = "public",
        group: str = "hermes-skills",
        output_dir: str,
    ) -> Dict[str, Any]:
        """Pull every skill in a namespace/group to ``output_dir``."""
        args = [
            "skill-sync",
            "--namespace", namespace,
            "--group", group,
            "--output", output_dir,
            "--json",
        ]
        return self._run(args)
