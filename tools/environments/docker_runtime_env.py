"""Freeze Docker env-file inputs before reuse selection and container launch."""

from pathlib import Path

from tools.environments.docker_egress import _env_flag_value


def snapshot_env_files(args: list[str]) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            break
        parsed = _env_flag_value(arg)
        index += 1
        if parsed is None:
            continue
        kind, value = parsed
        if value is None:
            value = args[index] if index < len(args) else ""
            index += 1
        if kind == "env-file" and value not in snapshots:
            try:
                snapshots[value] = Path(value).read_bytes()
            except OSError as exc:
                raise RuntimeError(f"Cannot read Docker env file {value!r}") from exc
    return snapshots


def stage_env_files(args: list[str], snapshots: dict[str, bytes], directory: Path) -> list[str]:
    """Replace only env-file operands; the fingerprint keeps their original paths."""
    staged = list(args)
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            break
        parsed = _env_flag_value(arg)
        index += 1
        if parsed is None:
            continue
        kind, value = parsed
        inline = value is not None
        if value is None:
            value = args[index] if index < len(args) else ""
            index += 1
        if kind != "env-file":
            continue
        path = directory / f"env-{index}"
        path.write_bytes(snapshots[value])
        if inline:
            staged[index - 1] = f"--env-file={path}"
        else:
            staged[index - 1] = str(path)
    return staged
