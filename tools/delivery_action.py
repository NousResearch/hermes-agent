"""Structured, role-scoped software-delivery effects.

No caller-provided shell command is executed on the host.  Verification commands
run in a networkless, hardened container with the source mounted read-only.
Git/GitHub lifecycle argv is constructed here and bound to the active immutable
policy; arbitrary REST, GraphQL, command flags, repositories, and work items are
not accepted.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import uuid
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

import psutil
from hermes_platform.resolver import LookupContext, locate_command
from tools.delivery_policy import DeliveryPolicy, current_delivery_policy
from tools.delivery_action_runtime import (
    DeliveryRuntimeError, require_free_disk, run_bounded, stream_to_file_bounded,
)
from tools.registry import registry


logger = logging.getLogger(__name__)
_SAFE_REF = re.compile(r"^[A-Za-z0-9._/-]+$")
_MAX_OUTPUT = 100_000
_GIT_TIMEOUT = 120
_NETWORK_TIMEOUT = 120
_VERIFY_TIMEOUT = 1800
_VERIFY_OUTPUT_LIMIT = 2 * 1024 * 1024
_TREE_OUTPUT_LIMIT = 32 * 1024 * 1024
_ARCHIVE_MAX_BYTES = 512 * 1024 * 1024
_EXTRACTED_MAX_BYTES = 1024 * 1024 * 1024
_SOURCE_MAX_FILES = 100_000
_DISK_RESERVE_BYTES = 64 * 1024 * 1024
_STALE_SECONDS = 60 * 60
_GH_PATH: Optional[str] = None


class DeliveryActionError(RuntimeError):
    pass


def _result(*, ok: bool, **values: Any) -> str:
    return json.dumps({"ok": ok, **values}, ensure_ascii=False)


def _trim(value: str) -> str:
    if len(value) <= _MAX_OUTPUT:
        return value
    return value[:_MAX_OUTPUT] + "\n...[output truncated by delivery_action]"


def _run(
    argv: list[str], *, cwd: Optional[str] = None, timeout: int, env: Optional[dict[str, str]] = None,
    output_limit: int = _MAX_OUTPUT,
) -> subprocess.CompletedProcess[str]:
    logger.info("delivery_action exec program=%s argc=%d cwd=%s", argv[0], len(argv), cwd)
    try:
        return run_bounded(
            argv, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit,
        )
    except DeliveryRuntimeError as exc:
        raise DeliveryActionError(str(exc)) from exc


def _git_argv(args: list[str]) -> list[str]:
    return [
        "git",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.fsmonitor=false",
        "-c", "commit.gpgSign=false",
        "-c", "protocol.ext.allow=never",
        "-c", "protocol.file.allow=never",
        "--literal-pathspecs",
        *args,
    ]


def _git_env(*, credentials: bool = True) -> dict[str, str]:
    from tools.environments.local import served_profile_child_env
    env = served_profile_child_env(inherit_credentials=credentials)
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    })
    return env


def _git(policy: DeliveryPolicy, args: list[str], *, timeout: int = _GIT_TIMEOUT) -> subprocess.CompletedProcess[str]:
    if not policy.workspace:
        raise DeliveryActionError("this operation requires a policy-bound workspace")
    return _run(
        _git_argv(args), cwd=policy.workspace, timeout=timeout, env=_git_env(credentials=True),
    )


def _git_ok(policy: DeliveryPolicy, args: list[str], *, timeout: int = _GIT_TIMEOUT) -> str:
    proc = _git(policy, args, timeout=timeout)
    if proc.returncode:
        raise DeliveryActionError(_trim(proc.stderr.strip() or proc.stdout.strip() or "git operation failed"))
    return _trim(proc.stdout)


def _trusted_gh_path() -> str:
    """Resolve gh once from a server-controlled search path, never a caller PATH."""
    global _GH_PATH
    if _GH_PATH is not None:
        return _GH_PATH
    search_path = os.pathsep.join(("/usr/local/bin", "/usr/bin", "/bin"))
    resolution = locate_command("gh", LookupContext(path=search_path))
    if not resolution.command:
        raise DeliveryActionError("GitHub CLI is unavailable; refusing the structured remote operation")
    candidate = resolution.command[0]
    resolved = str(Path(candidate).resolve(strict=True))
    try:
        mode = os.stat(resolved).st_mode
    except OSError as exc:
        raise DeliveryActionError("trusted GitHub CLI path is unavailable") from exc
    if not Path(resolved).is_absolute() or not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise DeliveryActionError("trusted GitHub CLI path is not an executable regular file")
    _GH_PATH = resolved
    return resolved


def _github_env() -> dict[str, str]:
    """Minimal github.com-only environment with the active profile's token."""
    from agent.secret_scope import get_secret

    token = get_secret("GH_TOKEN") or get_secret("GITHUB_TOKEN")
    if not token:
        raise DeliveryActionError(
            "the served profile has no GH_TOKEN or GITHUB_TOKEN for structured GitHub operations"
        )
    return {
        "GH_TOKEN": token,
        "GH_HOST": "github.com",
        "GH_PROMPT_DISABLED": "1",
        "GH_PAGER": "cat",
        "PAGER": "cat",
        "LANG": "C.UTF-8",
    }


def _gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return _run([_trusted_gh_path(), *args], timeout=_NETWORK_TIMEOUT, env=_github_env())


def _gh_json(endpoint: str, *, paginate: bool = False) -> Any:
    args = ["api", "--hostname", "github.com", "--method", "GET", endpoint]
    if paginate:
        args.append("--paginate")
        args.extend(["--slurp"])
    proc = _gh(args)
    if proc.returncode:
        raise DeliveryActionError(_trim(proc.stderr.strip() or "GitHub query failed"))
    try:
        payload = json.loads(proc.stdout or "null")
    except json.JSONDecodeError as exc:
        raise DeliveryActionError("GitHub returned malformed JSON") from exc
    if paginate and isinstance(payload, list) and payload and all(isinstance(page, list) for page in payload):
        return [item for page in payload for item in page]
    return payload


def _policy() -> DeliveryPolicy:
    policy = current_delivery_policy()
    if policy is None or policy.role is None:
        raise DeliveryActionError("delivery_action requires an immutable delivery-role context")
    return policy


def _only(args: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise DeliveryActionError(f"unsupported argument(s) for this structured action: {', '.join(unknown)}")


def _safe_ref(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result or not _SAFE_REF.fullmatch(result) or result.startswith(("-", "/")) or ".." in result.split("/"):
        raise DeliveryActionError(f"{field} is not a safe git ref/name")
    return result


def _safe_repo_path(value: Any, *, allow_dot: bool = False) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if allow_dot and raw in ("", "."):
        return "."
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ":" in raw or "\x00" in raw or any(part in ("", ".", "..") for part in path.parts):
        raise DeliveryActionError("path must be a relative repository path without traversal or pathspec magic")
    if ".git" in path.parts:
        raise DeliveryActionError("repository control files are not exposed")
    return str(path)


def _repo_from_url(value: str) -> Optional[str]:
    value = value.strip()
    if value.startswith("git@github.com:"):
        path = value.split(":", 1)[1]
    else:
        parsed = urlparse(value)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            return None
        path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
        return path.lower()
    return None


def _configured_repositories(policy: DeliveryPolicy) -> set[str]:
    output = _git_ok(policy, ["remote", "-v"])
    repositories: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and (repo := _repo_from_url(parts[1])):
            repositories.add(repo)
    return repositories


def _assert_configured_repository(policy: DeliveryPolicy, repository: str) -> None:
    if repository.lower() not in _configured_repositories(policy):
        raise DeliveryActionError(
            "repository is not one of the GitHub repositories configured as a remote in the bound workspace"
        )


def _assert_workspace_is_repository_root(policy: DeliveryPolicy) -> None:
    """Refuse git effects through an enclosing repository outside the bound root."""
    if not policy.workspace:
        raise DeliveryActionError("this operation requires a policy-bound workspace")
    root = _git_ok(policy, ["rev-parse", "--show-toplevel"]).strip()
    try:
        if Path(root).resolve(strict=True) != Path(policy.workspace).resolve(strict=True):
            raise DeliveryActionError("the policy workspace must be the repository top-level directory")
    except (OSError, RuntimeError) as exc:
        raise DeliveryActionError("the policy workspace repository root is unavailable") from exc


def _assert_pr_head(policy: DeliveryPolicy) -> dict[str, Any]:
    if not policy.repository or policy.pull_request is None or not policy.exact_sha:
        raise DeliveryActionError("this role has no policy-bound repository, pull request, and exact SHA")
    payload = _gh_json(f"repos/{policy.repository}/pulls/{policy.pull_request}")
    if not isinstance(payload, dict):
        raise DeliveryActionError("GitHub returned malformed pull-request metadata")
    actual = str((payload.get("head") or {}).get("sha") or "").lower()
    if actual != policy.exact_sha:
        raise DeliveryActionError(
            f"pull request head changed: expected {policy.exact_sha}, found {actual or 'unknown'}"
        )
    return payload


def _extract_archive_member(
    bundle: tarfile.TarFile, member: tarfile.TarInfo, destination: Path,
    relative: PurePosixPath, deadline: float,
) -> None:
    target = destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    cursor = destination
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise DeliveryActionError("exact-SHA archive traverses a symbolic link")
    if member.isdir():
        if target.exists() and not target.is_dir():
            raise DeliveryActionError("exact-SHA archive has conflicting entries")
        target.mkdir(exist_ok=True)
        return
    if member.issym():
        link = PurePosixPath(member.linkname)
        if link.is_absolute() or any(part == ".." for part in link.parts):
            raise DeliveryActionError("exact-SHA archive contains an escaping symbolic link")
        try:
            target.symlink_to(member.linkname)
        except OSError as exc:
            raise DeliveryActionError("exact-SHA archive contains an invalid symbolic link") from exc
        return
    if not member.isfile():
        raise DeliveryActionError(
            "exact-SHA archive contains an unsupported special or hard-linked entry"
        )
    source = bundle.extractfile(member)
    if source is None:
        raise DeliveryActionError("exact-SHA archive contains an unreadable file")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(target, flags, 0o555 if member.mode & 0o111 else 0o444)
    except OSError as exc:
        raise DeliveryActionError("exact-SHA archive has conflicting file entries") from exc
    with source, os.fdopen(fd, "wb") as output:
        remaining = member.size
        while remaining:
            if time.monotonic() >= deadline:
                raise DeliveryActionError("exact-SHA archive extraction timed out")
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                raise DeliveryActionError("exact-SHA archive file ended before its declared size")
            output.write(chunk)
            remaining -= len(chunk)


def _extract_git_archive(
    archive: Path, destination: Path, *, timeout: int = _GIT_TIMEOUT,
) -> None:
    """Materialize an archive without following links and under hard quotas."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise DeliveryActionError("safe exact-SHA materialization is unavailable on this platform")
    seen: set[PurePosixPath] = set()
    extracted_bytes = 0
    file_count = 0
    deadline = time.monotonic() + timeout
    with tarfile.open(archive, mode="r:") as bundle:
        for member in bundle:
            if time.monotonic() >= deadline:
                raise DeliveryActionError("exact-SHA archive extraction timed out")
            relative = PurePosixPath(member.name)
            if (
                not member.name or relative.is_absolute()
                or any(part in ("", ".", "..") for part in relative.parts)
                or relative in seen
            ):
                raise DeliveryActionError("exact-SHA archive contains an unsafe or duplicate path")
            seen.add(relative)
            file_count += 1
            if file_count > _SOURCE_MAX_FILES:
                raise DeliveryActionError("exact-SHA archive exceeds the source-file quota")
            if member.size < 0 or member.size > _EXTRACTED_MAX_BYTES - extracted_bytes:
                raise DeliveryActionError("exact-SHA archive exceeds the extracted-size quota")
            extracted_bytes += member.size
            try:
                require_free_disk(destination, member.size, reserve=_DISK_RESERVE_BYTES)
            except DeliveryRuntimeError as exc:
                raise DeliveryActionError(str(exc)) from exc
            _extract_archive_member(bundle, member, destination, relative, deadline)


def _assert_no_gitlinks(policy: DeliveryPolicy, exact_sha: str = "") -> None:
    args = ["ls-tree", "-r", "--full-tree", exact_sha] if exact_sha else ["ls-files", "--stage"]
    try:
        tree = run_bounded(
            _git_argv(args), cwd=policy.workspace, env=_git_env(credentials=False),
            timeout=_GIT_TIMEOUT, output_limit=_TREE_OUTPUT_LIMIT,
        )
    except DeliveryRuntimeError as exc:
        raise DeliveryActionError(str(exc)) from exc
    if tree.returncode:
        label = "exact-SHA tree" if exact_sha else "workspace index"
        raise DeliveryActionError(_trim(tree.stderr.strip() or f"{label} inspection failed"))
    if any(line.startswith("160000 ") for line in tree.stdout.splitlines()):
        raise DeliveryActionError(
            "delivery verification does not support git submodules; refusing gitlink materialization"
        )


def _exact_sha_source(policy: DeliveryPolicy, exact_sha: str, temp_root: Path) -> Path:
    """Stream exact git objects under quota; gitlinks fail closed as unsupported."""
    _assert_workspace_is_repository_root(policy)
    _assert_no_gitlinks(policy, exact_sha)
    try:
        require_free_disk(temp_root, 0, reserve=_DISK_RESERVE_BYTES)
    except DeliveryRuntimeError as exc:
        raise DeliveryActionError(str(exc)) from exc
    archive = temp_root / "source.tar"
    try:
        proc = stream_to_file_bounded(
            _git_argv(["archive", "--format=tar", exact_sha]), archive,
            cwd=policy.workspace, env=_git_env(credentials=False), timeout=_GIT_TIMEOUT,
            byte_limit=_ARCHIVE_MAX_BYTES, disk_reserve=_DISK_RESERVE_BYTES,
        )
    except DeliveryRuntimeError as exc:
        raise DeliveryActionError(str(exc)) from exc
    if proc.returncode:
        raise DeliveryActionError(_trim(proc.stderr.strip() or "exact-SHA archive failed"))
    source = temp_root / "checkout"
    source.mkdir(mode=0o755)
    _extract_git_archive(archive, source)
    archive.unlink()
    resolved = _git_ok(policy, ["rev-parse", "--verify", f"{exact_sha}^{{commit}}"]).strip().lower()
    if resolved != exact_sha:
        raise DeliveryActionError("policy-bound SHA does not resolve to the expected commit")
    return source


def _copy_workspace_file(
    entry: os.DirEntry[str], target: Path, destination: Path,
    quota: list[int], deadline: float,
) -> None:
    if time.monotonic() >= deadline:
        raise DeliveryActionError("workspace snapshot timed out")
    metadata = entry.stat(follow_symlinks=False)
    if metadata.st_size < 0 or metadata.st_size > _EXTRACTED_MAX_BYTES - quota[0]:
        raise DeliveryActionError("workspace snapshot exceeds the source-size quota")
    try:
        require_free_disk(destination, metadata.st_size, reserve=_DISK_RESERVE_BYTES)
    except DeliveryRuntimeError as exc:
        raise DeliveryActionError(str(exc)) from exc
    fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
    mode = 0o555 if metadata.st_mode & 0o111 else 0o444
    with os.fdopen(fd, "rb") as input_file:
        out_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(out_fd, "wb") as output_file:
            while chunk := input_file.read(64 * 1024):
                if time.monotonic() >= deadline:
                    raise DeliveryActionError("workspace snapshot timed out")
                if len(chunk) > _EXTRACTED_MAX_BYTES - quota[0]:
                    raise DeliveryActionError("workspace snapshot exceeds the source-size quota")
                try:
                    require_free_disk(destination, len(chunk), reserve=_DISK_RESERVE_BYTES)
                except DeliveryRuntimeError as exc:
                    raise DeliveryActionError(str(exc)) from exc
                quota[0] += len(chunk)
                output_file.write(chunk)


def _copy_workspace_tree(
    source: Path, destination: Path, quota: list[int], deadline: float,
) -> None:
    """Copy the live tree without VCS metadata, following no links and under quota."""
    destination.mkdir(mode=0o755)
    try:
        entries = os.scandir(source)
    except OSError as exc:
        raise DeliveryActionError(f"workspace snapshot is unreadable: {exc}") from exc
    with entries:
        for entry in entries:
            if time.monotonic() >= deadline:
                raise DeliveryActionError("workspace snapshot timed out")
            if entry.name == ".git":
                continue
            quota[1] += 1
            if quota[1] > _SOURCE_MAX_FILES:
                raise DeliveryActionError("workspace snapshot exceeds the source-file quota")
            target = destination / entry.name
            try:
                if entry.is_symlink():
                    target.symlink_to(os.readlink(entry.path))
                elif entry.is_dir(follow_symlinks=False):
                    _copy_workspace_tree(Path(entry.path), target, quota, deadline)
                elif entry.is_file(follow_symlinks=False):
                    _copy_workspace_file(entry, target, destination, quota, deadline)
                else:
                    raise DeliveryActionError("workspace contains an unsupported special file")
            except DeliveryActionError:
                raise
            except OSError as exc:
                raise DeliveryActionError(f"workspace snapshot failed at {entry.name!r}: {exc}") from exc


def _workspace_source(policy: DeliveryPolicy, temp_root: Path) -> Path:
    if not hasattr(os, "O_NOFOLLOW"):
        raise DeliveryActionError("safe workspace materialization is unavailable on this platform")
    try:
        require_free_disk(temp_root, 0, reserve=_DISK_RESERVE_BYTES)
    except DeliveryRuntimeError as exc:
        raise DeliveryActionError(str(exc)) from exc
    _assert_workspace_is_repository_root(policy)
    _assert_no_gitlinks(policy)
    source = temp_root / "checkout"
    _copy_workspace_tree(
        Path(policy.workspace), source, [0, 0], time.monotonic() + _GIT_TIMEOUT,
    )
    return source


def _docker_path() -> str:
    search_path = os.pathsep.join(("/usr/local/bin", "/usr/bin", "/bin"))
    resolution = locate_command("docker", LookupContext(path=search_path))
    if not resolution.command:
        raise DeliveryActionError(
            "sandboxed delivery verification requires Docker; refusing to run against the live checkout"
        )
    candidate = resolution.command[0]
    try:
        resolved = str(Path(candidate).resolve(strict=True))
        mode = os.stat(resolved).st_mode
    except OSError as exc:
        raise DeliveryActionError("trusted Docker executable path is unavailable") from exc
    if not Path(resolved).is_absolute() or not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise DeliveryActionError("trusted Docker path is not an executable regular file")
    return resolved


def _pid_is_live(pid: int) -> bool:
    return psutil.pid_exists(pid)


def _cleanup_stale_verification_state(docker: str) -> None:
    """Reap interrupted prior runs without touching a concurrent live verifier."""
    temp = Path(tempfile.gettempdir())
    now = time.time()
    uid_reader = getattr(os, "getuid", None)
    current_uid = uid_reader() if uid_reader is not None else None
    for entry in temp.glob("hermes-delivery-review-*-*"):
        try:
            if (
                not entry.is_dir() or entry.is_symlink()
                or (current_uid is not None and entry.stat().st_uid != current_uid)
            ):
                continue
            match = re.fullmatch(r"hermes-delivery-review-([0-9]+)-.+", entry.name)
            stale = now - entry.stat().st_mtime >= _STALE_SECONDS
            if stale or (match and not _pid_is_live(int(match.group(1)))):
                shutil.rmtree(entry)
        except OSError:
            logger.warning("delivery verification stale temp cleanup failed for %s", entry)
    try:
        listed = _run(
            [docker, "ps", "-a", "--filter", "label=com.hermes.delivery",
             "--format", '{{.ID}} {{.Label "com.hermes.delivery.pid"}}'],
            timeout=30, env={"PATH": "/usr/bin:/bin"},
        )
        for line in listed.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit() and not _pid_is_live(int(parts[1])):
                _run(
                    [docker, "rm", "--force", parts[0]], timeout=30,
                    env={"PATH": "/usr/bin:/bin"},
                )
    except DeliveryActionError as exc:
        logger.warning("delivery verification stale container cleanup failed: %s", exc)


def _docker_verify(
    policy: DeliveryPolicy, command: Any, *, exact_sha: str = "", acceptance: bool = False,
) -> str:
    recipe = policy.acceptance_recipe if acceptance else None
    if acceptance:
        if recipe is None:
            raise DeliveryActionError("closure policy has no server-bound acceptance recipe")
        command = list(recipe.argv)
        image_name = recipe.image
    else:
        image_name = policy.verification_image
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item and "\x00" not in item for item in command
    ):
        raise DeliveryActionError("command must be a non-empty array of argv strings")
    if len(command) > 128 or sum(len(item.encode("utf-8")) for item in command) > 32_768:
        raise DeliveryActionError("verification command is too large")
    if not image_name:
        raise DeliveryActionError(
            "verification is unavailable until delegation.delivery.verification_image is configured"
        )
    docker = _docker_path()
    _cleanup_stale_verification_state(docker)
    if not policy.workspace:
        raise DeliveryActionError("verification requires a policy-bound workspace")

    temp_root = Path(tempfile.mkdtemp(prefix=f"hermes-delivery-review-{os.getpid()}-"))
    try:
        source = (
            _exact_sha_source(policy, exact_sha, temp_root)
            if exact_sha else _workspace_source(policy, temp_root)
        )
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    container_name = f"hermes-delivery-{uuid.uuid4().hex}"
    docker_args = [
        docker, "run", "--rm", "--name", container_name,
        "--label", "com.hermes.delivery=verification",
        "--label", f"com.hermes.delivery.pid={os.getpid()}",
        "--pull", "never", "--entrypoint", "",
        "--network", "none", "--read-only", "--user", "65534:65534",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--memory", "4g", "--memory-swap", "4g", "--cpus", "2",
        "--ulimit", "nofile=1024:1024", "--ipc", "none",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777",  # no-tmp: ok — path is inside isolated container
        "--tmpfs", "/home:rw,nosuid,nodev,size=256m,mode=1777",
        "--env", "HOME=/home", "--env", "TMPDIR=/tmp",  # no-tmp: ok — container-only controlled tmpfs
        "--env", "XDG_CACHE_HOME=/tmp/cache",  # no-tmp: ok — container-only controlled cache
        "-v", f"{source}:/workspace:ro", "-w", "/workspace", image_name, *command,
    ]
    try:
        try:
            proc = run_bounded(
                docker_args, timeout=_VERIFY_TIMEOUT, output_limit=_VERIFY_OUTPUT_LIMIT,
                env={"PATH": "/usr/bin:/bin"},
            )
        except DeliveryRuntimeError as exc:
            raise DeliveryActionError(str(exc)) from exc
        return _result(
            ok=proc.returncode == 0,
            action="verify",
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exact_sha=exact_sha or None,
            attestation={
                "image": image_name,
                "recipe_identity": recipe.identity if recipe is not None else None,
                "entrypoint": "neutralized",
            },
            sandbox={
                "network": "none", "checkout": "read-only", "host_credentials": "not forwarded",
                "capabilities": "dropped", "image_pull": "disabled",
                "output_limit_bytes": _VERIFY_OUTPUT_LIMIT,
            },
        )
    finally:
        # A subprocess timeout kills only the Docker client. Force-removing the
        # named container prevents repository-controlled tests from surviving
        # the bounded verification call.
        try:
            _run(
                [docker, "rm", "--force", container_name], timeout=30,
                env={"PATH": "/usr/bin:/bin"},
            )
        except DeliveryActionError as exc:
            logger.warning("delivery verification container cleanup failed: %s", exc)
        shutil.rmtree(temp_root, ignore_errors=True)


def _implementer(policy: DeliveryPolicy, action: str, args: Mapping[str, Any]) -> str:
    if not policy.workspace:
        raise DeliveryActionError("implementer delivery_action requires a bound workspace")
    _assert_workspace_is_repository_root(policy)

    if action == "status":
        _only(args, {"action"})
        return _result(ok=True, action=action, output=_git_ok(policy, ["status", "--short", "--branch"]))
    if action == "diff":
        _only(args, {"action", "staged", "paths"})
        paths = args.get("paths") or []
        if not isinstance(paths, list):
            raise DeliveryActionError("paths must be an array")
        normalized = [_safe_repo_path(path) for path in paths]
        argv = ["diff", "--no-ext-diff", "--no-color"]
        if args.get("staged"):
            argv.append("--cached")
        if normalized:
            argv.extend(["--", *normalized])
        return _result(ok=True, action=action, output=_git_ok(policy, argv))
    if action == "stage":
        _only(args, {"action", "paths"})
        paths = args.get("paths")
        if not isinstance(paths, list) or not paths:
            raise DeliveryActionError("stage requires a non-empty paths array")
        normalized = [_safe_repo_path(path) for path in paths]
        _git_ok(policy, ["add", "--", *normalized])
        return _result(ok=True, action=action, paths=normalized)
    if action == "commit":
        _only(args, {"action", "message"})
        message = str(args.get("message") or "").strip()
        if not message or len(message) > 10_000 or "\x00" in message:
            raise DeliveryActionError("commit requires a non-empty message no longer than 10,000 characters")
        _git_ok(policy, ["commit", "--no-verify", "-m", message])
        sha = _git_ok(policy, ["rev-parse", "HEAD"]).strip()
        return _result(ok=True, action=action, sha=sha)
    if action == "push":
        _only(args, {"action", "remote", "branch"})
        remote = _safe_ref(args.get("remote") or "origin", "remote")
        branch = _safe_ref(args.get("branch"), "branch")
        _git_ok(policy, ["remote", "get-url", remote])
        _git_ok(policy, ["push", "--porcelain", remote, f"HEAD:refs/heads/{branch}"], timeout=300)
        return _result(ok=True, action=action, remote=remote, branch=branch)
    if action == "open_pr":
        _only(args, {"action", "repository", "base", "title", "body", "draft"})
        repository = str(args.get("repository") or "").strip()
        _assert_configured_repository(policy, repository)
        base = _safe_ref(args.get("base") or "main", "base")
        title = str(args.get("title") or "").strip()
        body = str(args.get("body") or "")
        if not title or len(title) > 256 or "\x00" in title or "\x00" in body:
            raise DeliveryActionError("open_pr requires a valid non-empty title")
        branch = _git_ok(policy, ["branch", "--show-current"]).strip()
        if not branch:
            raise DeliveryActionError("cannot open a pull request from a detached HEAD")
        argv = [
            "pr", "create", "--repo", repository, "--base", base, "--head", branch,
            "--title", title, "--body", body,
        ]
        if args.get("draft"):
            argv.append("--draft")
        proc = _gh(argv)
        if proc.returncode:
            raise DeliveryActionError(_trim(proc.stderr.strip() or "pull-request creation failed"))
        return _result(ok=True, action=action, url=proc.stdout.strip())
    if action == "verify":
        _only(args, {"action", "command"})
        return _docker_verify(policy, args.get("command"))
    raise DeliveryActionError(f"action {action!r} is not available to implementer")


def _reviewer(policy: DeliveryPolicy, action: str, args: Mapping[str, Any]) -> str:
    _assert_workspace_is_repository_root(policy)
    _assert_configured_repository(policy, policy.repository)
    if action == "pr_status":
        _only(args, {"action"})
        pr = _assert_pr_head(policy)
        return _result(
            ok=True,
            action=action,
            repository=policy.repository,
            pull_request=policy.pull_request,
            exact_sha=policy.exact_sha,
            state=pr.get("state"),
            draft=bool(pr.get("draft")),
            title=pr.get("title"),
        )
    if action == "read_file":
        _only(args, {"action", "path"})
        _assert_pr_head(policy)
        path = _safe_repo_path(args.get("path"))
        output = _git_ok(policy, ["show", f"{policy.exact_sha}:{path}"])
        return _result(ok=True, action=action, path=path, exact_sha=policy.exact_sha, content=output)
    if action == "search":
        _only(args, {"action", "pattern", "path"})
        _assert_pr_head(policy)
        pattern = str(args.get("pattern") or "")
        if not pattern or len(pattern) > 2_000 or "\x00" in pattern:
            raise DeliveryActionError("search requires a non-empty pattern no longer than 2,000 characters")
        path = _safe_repo_path(args.get("path") or ".", allow_dot=True)
        proc = _git(policy, ["grep", "-n", "--no-color", "-e", pattern, policy.exact_sha, "--", path])
        if proc.returncode not in (0, 1):
            raise DeliveryActionError(_trim(proc.stderr.strip() or "git search failed"))
        return _result(ok=True, action=action, exact_sha=policy.exact_sha, matches=_trim(proc.stdout))
    if action == "verify":
        _only(args, {"action", "command"})
        _assert_pr_head(policy)
        return _docker_verify(policy, args.get("command"), exact_sha=policy.exact_sha)
    raise DeliveryActionError(f"action {action!r} is not available to reviewer")


def _classic_protection_receipt(protection: Any) -> Optional[dict[str, Any]]:
    if not isinstance(protection, dict):
        return None
    reviews = protection.get("required_pull_request_reviews")
    checks = protection.get("required_status_checks")
    admins = protection.get("enforce_admins")
    if not isinstance(reviews, dict) or not isinstance(checks, dict):
        return None
    required_checks = checks.get("checks") or checks.get("contexts") or []
    admin_enforced = admins.get("enabled") if isinstance(admins, dict) else admins
    if not (
        isinstance(reviews.get("required_approving_review_count"), int)
        and not isinstance(reviews.get("required_approving_review_count"), bool)
        and reviews["required_approving_review_count"] >= 1
        and reviews.get("dismiss_stale_reviews") is True
        and reviews.get("require_last_push_approval") is True
        and checks.get("strict") is True
        and isinstance(required_checks, list) and required_checks
        and admin_enforced is True
    ):
        return None
    return {"kind": "branch_protection", "required_checks": len(required_checks)}


def _ruleset_protection_receipt(repository: str, branch: str) -> Optional[dict[str, Any]]:
    try:
        rules = _gh_json(
            f"repos/{repository}/rules/branches/{branch}?per_page=100", paginate=True,
        )
    except DeliveryActionError:
        return None
    if not isinstance(rules, list):
        return None
    pull_rules = [r for r in rules if isinstance(r, dict) and r.get("type") == "pull_request"]
    check_rules = [r for r in rules if isinstance(r, dict) and r.get("type") == "required_status_checks"]
    pull_ok = any(
        isinstance(rule.get("parameters"), dict)
        and isinstance(rule["parameters"].get("required_approving_review_count"), int)
        and not isinstance(rule["parameters"].get("required_approving_review_count"), bool)
        and rule["parameters"]["required_approving_review_count"] >= 1
        and rule["parameters"].get("dismiss_stale_reviews_on_push") is True
        and rule["parameters"].get("require_last_push_approval") is True
        for rule in pull_rules
    )
    checks_ok = any(
        isinstance(rule.get("parameters"), dict)
        and bool(rule["parameters"].get("required_status_checks"))
        and rule["parameters"].get("strict_required_status_checks_policy") is True
        for rule in check_rules
    )
    ids = {
        rule.get("ruleset_id") for rule in (*pull_rules, *check_rules)
        if isinstance(rule.get("ruleset_id"), int)
    }
    if not pull_ok or not checks_ok or not ids:
        return None
    for ruleset_id in ids:
        details = _gh_json(f"repos/{repository}/rulesets/{ruleset_id}")
        if (
            not isinstance(details, dict) or details.get("enforcement") != "active"
            or details.get("bypass_actors") != []
        ):
            return None
    return {"kind": "ruleset", "rulesets": len(ids)}


def _assert_merge_protection(repository: str, branch: str) -> dict[str, Any]:
    """Prove GitHub itself will enforce the exact-review/check contract."""
    classic = None
    try:
        classic = _classic_protection_receipt(
            _gh_json(f"repos/{repository}/branches/{branch}/protection")
        )
    except DeliveryActionError:
        pass
    receipt = classic or _ruleset_protection_receipt(repository, branch)
    if receipt is None:
        raise DeliveryActionError(
            "repository protection does not provably enforce stale-review dismissal, independent last-push approval, "
            "strict required checks, and administrator enforcement"
        )
    return receipt


def _merger(policy: DeliveryPolicy, action: str, args: Mapping[str, Any]) -> str:
    if action != "merge":
        raise DeliveryActionError("merger may request only the structured merge action")
    _only(args, {"action"})
    pr = _assert_pr_head(policy)
    if pr.get("state") != "open" or pr.get("draft"):
        raise DeliveryActionError("pull request must be open and non-draft")
    base_branch = str((pr.get("base") or {}).get("ref") or "")
    if not base_branch:
        raise DeliveryActionError("pull request base branch is unavailable")

    reviews = _gh_json(f"repos/{policy.repository}/pulls/{policy.pull_request}/reviews?per_page=100", paginate=True)
    if not isinstance(reviews, list):
        raise DeliveryActionError("GitHub returned malformed review metadata")
    author = str((pr.get("user") or {}).get("login") or "").lower()
    latest: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict):
            continue
        login = str((review.get("user") or {}).get("login") or "").lower()
        if login:
            latest[login] = review
    approvals = [
        review for login, review in latest.items()
        if login != author
        and str(review.get("state") or "").upper() == "APPROVED"
        and str(review.get("commit_id") or "").lower() == policy.exact_sha
    ]
    if not approvals:
        raise DeliveryActionError(
            "no current independent GitHub approval is bound to the policy exact PR head"
        )

    checks_proc = _gh([
        "pr", "checks", str(policy.pull_request), "--repo", policy.repository,
        "--required", "--json", "name,bucket,state,workflow",
    ])
    if checks_proc.returncode:
        raise DeliveryActionError(_trim(checks_proc.stderr.strip() or "required CI query failed"))
    try:
        checks = json.loads(checks_proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise DeliveryActionError("required CI query returned malformed JSON") from exc
    if not isinstance(checks, list) or not checks:
        raise DeliveryActionError("no required CI checks were reported; refusing merge")
    failing = [
        str(check.get("name") or "unknown") for check in checks
        if not isinstance(check, dict) or str(check.get("bucket") or "").lower() not in {"pass", "skipping"}
    ]
    if failing:
        raise DeliveryActionError(f"required CI is not passing: {', '.join(failing)}")

    protection = _assert_merge_protection(policy.repository, base_branch)
    # Re-read the target after every evidence query. GitHub also atomically
    # rejects a head race at merge time via --match-head-commit.
    _assert_pr_head(policy)
    # Keep repository-side enforcement as the final remote observation too;
    # if it changed during evidence collection, fail before issuing merge.
    protection = _assert_merge_protection(policy.repository, base_branch)
    merge_proc = _gh([
        "pr", "merge", str(policy.pull_request), "--repo", policy.repository,
        "--squash", "--match-head-commit", policy.exact_sha,
    ])
    if merge_proc.returncode:
        raise DeliveryActionError(_trim(merge_proc.stderr.strip() or "merge failed"))
    return _result(
        ok=True,
        action="merge",
        repository=policy.repository,
        pull_request=policy.pull_request,
        exact_sha=policy.exact_sha,
        required_checks=len(checks),
        independent_approvals=len(approvals),
        protection=protection,
        output=_trim(merge_proc.stdout),
    )


def _assert_merged_on_default(policy: DeliveryPolicy) -> str:
    repository = _gh_json(f"repos/{policy.repository}")
    if not isinstance(repository, dict) or not repository.get("default_branch"):
        raise DeliveryActionError("GitHub returned malformed repository metadata")
    default_branch = str(repository["default_branch"])
    comparison = _gh_json(
        f"repos/{policy.repository}/compare/{policy.merged_sha}...{default_branch}"
    )
    if not isinstance(comparison, dict) or comparison.get("status") not in {"ahead", "identical"}:
        raise DeliveryActionError("merged SHA is not present on the repository default branch")
    return default_branch


def _closure(policy: DeliveryPolicy, action: str, args: Mapping[str, Any]) -> str:
    if action != "close_issue":
        raise DeliveryActionError("closure controller may request only close_issue")
    _only(args, {"action"})
    if not policy.repository or policy.issue is None or not policy.merged_sha:
        raise DeliveryActionError("closure policy is missing repository, issue, or merged SHA")
    if policy.acceptance_recipe is None:
        raise DeliveryActionError("closure policy is missing its server-bound acceptance recipe")
    _assert_workspace_is_repository_root(policy)
    _assert_configured_repository(policy, policy.repository)

    _assert_merged_on_default(policy)
    verification = json.loads(
        _docker_verify(policy, None, exact_sha=policy.merged_sha, acceptance=True)
    )
    if not verification.get("ok"):
        raise DeliveryActionError("post-merge acceptance failed; refusing closure")
    # Containment is time-sensitive: revalidate immediately after acceptance.
    default_branch = _assert_merged_on_default(policy)

    issue = _gh_json(f"repos/{policy.repository}/issues/{policy.issue}")
    if not isinstance(issue, dict) or "pull_request" in issue:
        raise DeliveryActionError("policy-bound work item is not an issue")
    if issue.get("state") == "closed":
        return _result(
            ok=True, action=action, already_complete=True,
            repository=policy.repository, issue=policy.issue,
            merged_sha=policy.merged_sha, default_branch=default_branch,
            acceptance=verification.get("attestation"),
        )
    if issue.get("state") != "open":
        raise DeliveryActionError("policy-bound work item has an unsupported state")

    # The issue query can take time; make containment the final read before close.
    default_branch = _assert_merged_on_default(policy)
    recipe = policy.acceptance_recipe
    proc = _gh([
        "issue", "close", str(policy.issue), "--repo", policy.repository,
        "--comment", (
            f"Post-merge acceptance passed for {policy.merged_sha} "
            f"({recipe.identity})."
        ),
    ])
    if proc.returncode:
        raise DeliveryActionError(_trim(proc.stderr.strip() or "issue closure failed"))
    return _result(
        ok=True,
        action=action,
        already_complete=False,
        repository=policy.repository,
        issue=policy.issue,
        merged_sha=policy.merged_sha,
        default_branch=default_branch,
        acceptance=verification.get("attestation"),
        output=_trim(proc.stdout),
    )


def delivery_action(
    action: str,
    command: Optional[list[str]] = None,
    paths: Optional[list[str]] = None,
    staged: Optional[bool] = None,
    message: Optional[str] = None,
    remote: Optional[str] = None,
    branch: Optional[str] = None,
    repository: Optional[str] = None,
    base: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    draft: Optional[bool] = None,
    path: Optional[str] = None,
    pattern: Optional[str] = None,
) -> str:
    supplied = {
        key: value for key, value in {
            "action": action, "command": command, "paths": paths,
            "staged": staged, "message": message, "remote": remote, "branch": branch,
            "repository": repository, "base": base, "title": title, "body": body,
            "draft": draft, "path": path, "pattern": pattern,
        }.items() if value is not None
    }
    try:
        policy = _policy()
        normalized = str(action or "").strip().lower()
        if not normalized:
            raise DeliveryActionError("action is required")
        if policy.role == "implementer":
            return _implementer(policy, normalized, supplied)
        if policy.role == "reviewer":
            return _reviewer(policy, normalized, supplied)
        if policy.role == "merger":
            return _merger(policy, normalized, supplied)
        return _closure(policy, normalized, supplied)
    except DeliveryActionError as exc:
        logger.warning("delivery_action denied: %s", exc)
        return _result(ok=False, error=str(exc))


DELIVERY_ACTION_SCHEMA = {
    "name": "delivery_action",
    "description": (
        "Perform one structured operation authorized by the immutable delivery role. Host shell, arbitrary network/API, "
        "GraphQL, MCP, lifecycle flags, and caller-selected merger/closure targets are not accepted. Verification runs "
        "without network or host credentials against a read-only checkout."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "diff", "stage", "commit", "push", "open_pr", "verify",
                    "pr_status", "read_file", "search", "merge", "close_issue",
                ],
            },
            "command": {"type": "array", "items": {"type": "string"}},
            "paths": {"type": "array", "items": {"type": "string"}},
            "staged": {"type": "boolean"},
            "message": {"type": "string"},
            "remote": {"type": "string"},
            "branch": {"type": "string"},
            "repository": {"type": "string"},
            "base": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "draft": {"type": "boolean"},
            "path": {"type": "string"},
            "pattern": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

def delivery_action_handler(args: Mapping[str, Any], **_kwargs: Any) -> str:
    """Registry adapter kept explicit so the public operation remains keyword-only."""
    if not isinstance(args, Mapping):
        return json.dumps({"ok": False, "error": "delivery_action arguments must be an object"})
    supported = set(DELIVERY_ACTION_SCHEMA["parameters"]["properties"])
    unknown = sorted(set(args) - supported)
    if unknown:
        return json.dumps({
            "ok": False,
            "error": "unsupported argument(s): " + ", ".join(unknown),
        })
    return delivery_action(**dict(args))


registry.register(
    name="delivery_action",
    toolset="delivery",
    schema=DELIVERY_ACTION_SCHEMA,
    handler=delivery_action_handler,
)
