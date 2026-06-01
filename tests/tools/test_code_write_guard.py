import json

import pytest


@pytest.mark.parametrize(
    "path",
    [
        "app.py",
        "component.tsx",
        "styles.scss",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "tsconfig.json",
        "tsconfig.app.json",
        "vite.config.ts",
        "webpack.config.js",
        "next.config.mjs",
        "pyproject.toml",
        "poetry.lock",
        "requirements.txt",
        "requirements-dev.txt",
        "go.mod",
        "go.sum",
        "Cargo.toml",
        "Cargo.lock",
        "Dockerfile",
        "Containerfile",
        "docker-compose.yml",
        "docker-compose.override.yaml",
    ],
)
def test_code_write_guard_classifies_product_code_paths(monkeypatch, tmp_path, path):
    import tools.file_tools as file_tools

    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    assert file_tools._is_code_write_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "RESULT.md",
        "phase_RESULT.md",
        "evidence_manifest.json",
        "phase_evidence_manifest_v2.json",
        "run.log",
        "README.md",
        "notes.md",
        "SOUL.md",
        "CODEX.md",
        "config.yaml",
        "config.yml",
    ],
)
def test_code_write_guard_classifies_non_product_artifacts(monkeypatch, tmp_path, path):
    import tools.file_tools as file_tools

    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    assert not file_tools._is_code_write_path(path)


def test_code_write_guard_blocks_python_write_when_config_enabled(monkeypatch, tmp_path):
    import tools.file_tools as file_tools

    monkeypatch.setattr(file_tools, "_SENSITIVE_PATH_PREFIXES", ("/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"code_writes_require_codex": True},
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = json.loads(file_tools.write_file_tool("feature.py", "print('native')"))

    assert "error" in result
    assert "code_writes_require_codex" in result["error"]
    assert not (tmp_path / "feature.py").exists()


def test_code_write_guard_allows_markdown_result_when_enabled(monkeypatch, tmp_path):
    import tools.file_tools as file_tools

    monkeypatch.setattr(file_tools, "_SENSITIVE_PATH_PREFIXES", ("/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"code_writes_require_codex": True},
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = json.loads(file_tools.write_file_tool("RESULT.md", "ready for verification"))

    assert result.get("error") in (None, False)
    assert (tmp_path / "RESULT.md").read_text() == "ready for verification"


def test_code_write_guard_blocks_package_manifest_when_config_enabled(monkeypatch, tmp_path):
    import tools.file_tools as file_tools

    monkeypatch.setattr(file_tools, "_SENSITIVE_PATH_PREFIXES", ("/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"code_writes_require_codex": True},
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = json.loads(file_tools.write_file_tool("package.json", "{}\n"))

    assert "error" in result
    assert "code_writes_require_codex" in result["error"]
    assert not (tmp_path / "package.json").exists()


@pytest.mark.parametrize("path", ["evidence_manifest.json", "phase_RESULT.md", "config.yaml"])
def test_code_write_guard_allows_reporting_and_profile_artifacts_when_enabled(
    monkeypatch,
    tmp_path,
    path,
):
    import tools.file_tools as file_tools

    monkeypatch.setattr(file_tools, "_SENSITIVE_PATH_PREFIXES", ("/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"code_writes_require_codex": True},
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    target = tmp_path / path
    result = json.loads(file_tools.write_file_tool(str(target), "artifact\n"))

    assert result.get("error") in (None, False)
    assert target.read_text() == "artifact\n"


def test_code_write_guard_blocks_v4a_code_patch_when_config_enabled(monkeypatch, tmp_path):
    import tools.file_tools as file_tools

    target = tmp_path / "feature.py"
    target.write_text("old = 1\n")
    monkeypatch.setattr(file_tools, "_SENSITIVE_PATH_PREFIXES", ("/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"code_writes_require_codex": {"enabled": True}},
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = json.loads(
        file_tools.patch_tool(
            mode="patch",
            patch="*** Begin Patch\n*** Update File: feature.py\n@@\n-old = 1\n+old = 2\n*** End Patch",
        )
    )

    assert "error" in result
    assert "delegate_to_codex" in result["error"]
    assert target.read_text() == "old = 1\n"
