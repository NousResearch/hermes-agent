"""Safe dedicated PDF compression tool for restricted bot profiles.

This tool intentionally avoids shell execution and does not inherit the agent's
secret-filled environment. It is meant for narrow Telegram bots that should only
compress user-provided PDFs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.registry import registry


DEFAULT_MAX_INPUT_MB = 750.0
DEFAULT_MAX_TARGET_MB = 200.0
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_CPU_SECONDS = 1200
DEFAULT_MEMORY_GB = 5


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


PDF_COMPRESS_SCHEMA = {
    "name": "compress_pdf",
    "description": (
        "Compress a user-provided PDF to a target size using the profile's "
        "pdf-compress skill. This is a restricted wrapper: PDF input only, no "
        "shell, no network, no inherited API secrets. Returns an output PDF path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "Absolute local path to the uploaded PDF file.",
            },
            "target_mb": {
                "type": "number",
                "description": "Desired target size in MB. Defaults to 20. Allowed range: 1..200 by default.",
            },
        },
        "required": ["input_path"],
    },
}


def _json_error(message: str, **extra) -> str:
    payload = {"success": False, "error": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _is_pdf(path: Path) -> bool:
    if path.suffix.lower() != ".pdf":
        return False
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def _safe_filename(path: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in path.stem)
    stem = stem[:80].strip("._") or "document"
    return f"{stem}.pdf"


def _skill_paths(home: Path) -> tuple[Path, Path]:
    skill_dir = home / "skills" / "utilities" / "pdf-compress"
    script = skill_dir / "scripts" / "pdf_compress.py"
    venv_python = skill_dir / ".venv" / "bin" / "python"
    python = venv_python if venv_python.exists() else Path(sys.executable)
    return script, python


def _preexec_limits():
    """Best-effort resource limits for the child process (Linux/Unix)."""
    try:
        import resource

        cpu_seconds = _env_int("HERMES_PDF_COMPRESS_CPU_SECONDS", DEFAULT_CPU_SECONDS)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 60))
        # Large presentation PDFs can legitimately need several GiB while still
        # being bounded enough to prevent silly DoS cases.
        mem_gb = _env_int("HERMES_PDF_COMPRESS_MEMORY_GB", DEFAULT_MEMORY_GB)
        mem = mem_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        # 2 GiB max output file size.
        fsize = 2 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    except Exception:
        pass


def compress_pdf(input_path: str, target_mb: float | int | None = None) -> str:
    home = Path(get_hermes_home()).resolve()
    source_raw = Path(str(input_path)).expanduser()

    try:
        source = source_raw.resolve(strict=True)
    except FileNotFoundError:
        return _json_error("PDF file not found")
    except OSError as exc:
        return _json_error(f"Invalid PDF path: {exc}")

    if not source.is_file():
        return _json_error("Input is not a file")
    if not _is_pdf(source):
        return _json_error("Мне нужен PDF-файл.")

    size_mb = source.stat().st_size / (1024 * 1024)
    max_input_mb = _env_float("HERMES_PDF_MAX_INPUT_MB", DEFAULT_MAX_INPUT_MB)
    if size_mb > max_input_mb:
        return _json_error(
            "PDF is too large for safe compression",
            input_size_mb=round(size_mb, 2),
            max_mb=round(max_input_mb, 2),
        )

    try:
        target = 20.0 if target_mb is None else float(target_mb)
    except (TypeError, ValueError):
        return _json_error("Invalid target size")
    max_target_mb = _env_float("HERMES_PDF_MAX_TARGET_MB", DEFAULT_MAX_TARGET_MB)
    if not (1 <= target <= max_target_mb):
        return _json_error(f"Target size must be between 1 and {max_target_mb:g} MB")

    script, python = _skill_paths(home)
    if not script.exists():
        return _json_error("pdf-compress skill script not found", expected=str(script))
    if not python.exists():
        return _json_error("Python interpreter for pdf-compress not found", expected=str(python))

    jobs_dir = home / "pdf_jobs"
    jobs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    job_dir = jobs_dir / uuid.uuid4().hex
    job_dir.mkdir(mode=0o700)

    local_input = job_dir / _safe_filename(source)
    output = job_dir / f"{local_input.stem} (compressed).pdf"
    shutil.copy2(source, local_input)

    if size_mb <= target:
        shutil.copy2(local_input, output)
        return json.dumps(
            {
                "success": True,
                "input_path": str(source),
                "output_path": str(output),
                "input_size_mb": round(size_mb, 2),
                "output_size_mb": round(output.stat().st_size / (1024 * 1024), 2),
                "target_mb": target,
                "message": "Input is already under target size. Send it to the user by including MEDIA:<output_path> in the final response.",
            },
            ensure_ascii=False,
        )

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(home),
        "HERMES_HOME": str(home),
        "TMPDIR": str(job_dir),
    }

    cmd = [str(python), str(script), str(local_input), "--target", str(target), "--output", str(output)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_env_int("HERMES_PDF_COMPRESS_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
            preexec_fn=_preexec_limits if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired:
        return _json_error("Compression timed out", input_size_mb=round(size_mb, 2))

    stdout = (proc.stdout or "")[-4000:]
    stderr = (proc.stderr or "")[-4000:]
    if proc.returncode != 0 or not output.exists():
        return _json_error(
            "Compression failed",
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    output_size_mb = output.stat().st_size / (1024 * 1024)
    return json.dumps(
        {
            "success": True,
            "input_path": str(source),
            "output_path": str(output),
            "input_size_mb": round(size_mb, 2),
            "output_size_mb": round(output_size_mb, 2),
            "target_mb": target,
            "message": "Send the compressed PDF to the user by including MEDIA:<output_path> in the final response.",
            "stdout": stdout,
        },
        ensure_ascii=False,
    )


registry.register(
    name="compress_pdf",
    toolset="pdf_compress",
    schema=PDF_COMPRESS_SCHEMA,
    handler=lambda args, **kw: compress_pdf(
        input_path=args.get("input_path", ""),
        target_mb=args.get("target_mb"),
    ),
    check_fn=lambda: (Path(get_hermes_home()) / "skills" / "utilities" / "pdf-compress" / "scripts" / "pdf_compress.py").exists(),
    description="Restricted PDF compression wrapper",
    emoji="🗜️",
    max_result_size_chars=6000,
)
