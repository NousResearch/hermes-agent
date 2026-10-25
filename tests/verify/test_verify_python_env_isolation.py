"""verify must never let a project's pip touch the Hermes generation venv.

Regression: Hermes exports PYTHONPATH=<repo>:<generation site-packages>. A
``hermes verify`` bootstrap (``pip install -r requirements.txt``) inherited it, so pip
uninstalled mcp 2.x from the generation and installed the project's mcp<2 - every
HTTP MCP server then failed with "mcp.client.streamable_http is not available".
"""

import sys
import zipfile
from pathlib import Path

import pytest

from agent.verify import runner
from agent.verify.recipes import Recipe
from agent.verify.runner import run_verify


def _fake_installed_dist(site: Path, name: str, version: str) -> None:
    (site / name).mkdir(parents=True)
    (site / name / "__init__.py").write_text(f"__version__ = {version!r}\n")
    info = site / f"{name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (info / "INSTALLER").write_text("pip\n")
    (info / "RECORD").write_text(
        f"{name}/__init__.py,,\n{info.name}/METADATA,,\n{info.name}/INSTALLER,,\n{info.name}/RECORD,,\n"
    )


def _fake_wheel(dest: Path, name: str, version: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    info = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(dest / f"{name}-{version}-py3-none-any.whl", "w") as whl:
        whl.writestr(f"{name}/__init__.py", f"__version__ = {version!r}\n")
        whl.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        whl.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        whl.writestr(f"{info}/RECORD", f"{name}/__init__.py,,\n{info}/METADATA,,\n{info}/WHEEL,,\n{info}/RECORD,,\n")


def test_clean_child_env_drops_interpreter_steering(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/gen/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/gen")
    monkeypatch.setenv("VIRTUAL_ENV", "/gen/venv")
    env = runner._clean_child_env()
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & env.keys()

    venv = tmp_path / ".venv"
    env = runner._clean_child_env(venv)
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].split(":" if sys.platform != "win32" else ";")[0] == str(runner._venv_python(venv).parent)


@pytest.mark.parametrize("command", [
    "pip install -r requirements.txt", "pip3 install -e .", "python -m pip install x",
    "cd sub && pip install -r requirements.txt",
])
def test_pip_install_rewritten_to_isolated_venv_pip(command, tmp_path):
    venv = tmp_path / ".venv"
    out = runner._isolate_pip(command, venv)
    assert f"{runner._venv_python(venv)} -m pip install --isolated" in out
    assert not out.lstrip().startswith(("pip ", "pip3 "))


def test_non_pip_commands_untouched(tmp_path):
    assert runner._isolate_pip("pytest -q", tmp_path / ".venv") == "pytest -q"


def test_verify_bootstrap_leaves_generation_mcp_intact(monkeypatch, tmp_path):
    """A requirements.txt pinning mcp<2 must install into the project venv only."""
    generation = tmp_path / "generation" / "site-packages"
    _fake_installed_dist(generation, "mcp", "2.0.0")
    wheels = tmp_path / "wheels"
    _fake_wheel(wheels, "mcp", "1.0.0")

    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text(f"--no-index\n--find-links {wheels}\nmcp<2\n")
    monkeypatch.setenv("PYTHONPATH", str(generation))
    monkeypatch.setenv("PYTHONHOME", sys.base_prefix)

    recipe = Recipe(name="Python project", kind="python", bootstrap=["pip install -r requirements.txt"])
    result = run_verify(project, recipe, phases=["bootstrap"], skip_start=True, phase_timeout=300)

    assert result.ok, result.to_dict()
    # The generation still carries mcp 2.0.0, untouched.
    assert (generation / "mcp-2.0.0.dist-info" / "METADATA").is_file()
    assert (generation / "mcp" / "__init__.py").read_text() == "__version__ = '2.0.0'\n"
    assert not list(generation.glob("mcp-1.*"))
    # The project's pin landed in the project's own venv.
    assert list((project / ".venv").rglob("mcp-1.0.0.dist-info"))
    assert "--isolated" in result.phases[0].command
