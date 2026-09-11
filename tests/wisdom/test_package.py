import json
import os
from pathlib import Path

import pytest

import hermes_wisdom.package as package_module
from hermes_wisdom.package import (
    PackagePolicyError,
    infer_authoring_system_specification,
    prepare_package,
    verify_content_files,
)


def make_skill(root: Path) -> Path:
    skill = root / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Test.\n---\n\n# Test\n", encoding="utf-8"
    )
    refs = skill / "refs"
    refs.mkdir()
    (refs / "notes.txt").write_text("Exact notes.\n", encoding="utf-8")
    return skill


def test_preparation_creates_only_instruction_overlay_and_hashes(tmp_path: Path):
    skill = make_skill(tmp_path)
    package = prepare_package(
        skill,
        overlay_root=tmp_path / "overlays",
        author_description="  <b>Does</b> the useful thing. \r\n",
        owner="owner",
        installation_id="installation-123456",
    )
    assert package.description == "Does the useful thing."
    assert {item.path for item in package.files} == {
        "SKILL.md",
        "refs/notes.txt",
        "skill.manifest.json",
    }
    manifest = json.loads(
        (package.overlay / "skill.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["requirements"]["tools"] == []
    from hermes_cli import __version__ as hermes_version

    assert manifest["requirements"]["hermes"]["minimum_version"] == hermes_version
    assert len(manifest["requirements"]["platforms"]) == 1
    assert len(manifest["requirements"]["architectures"]) == 1
    assert package.content_hash.startswith("sha256:")


def test_authoring_defaults_normalize_device_and_explicit_skill_requirements(
    tmp_path: Path,
):
    skill = make_skill(tmp_path)
    (skill / "SKILL.md").write_text(
        """---
name: my-skill
description: Test.
metadata:
  hermes:
    requires_toolsets: [terminal, browser]
    requires_tools: [execute_code, terminal]
    requires_plugins:
      - local/example-plugin
      - id: local/optional-plugin
        minimum_version: 2.4.0
        required: false
---

# Test
""",
        encoding="utf-8",
    )

    specification = infer_authoring_system_specification(
        skill,
        hermes_version="9.8.7",
        system_name="Darwin",
        machine="aarch64",
    )

    assert specification.hermes.minimum_version == "9.8.7"
    assert specification.platforms == ["macOS"]
    assert specification.architectures == ["arm64"]
    assert [requirement.name for requirement in specification.tools] == [
        "terminal",
        "browser",
        "execute_code",
    ]
    assert all(requirement.auto_install is False for requirement in specification.tools)
    assert specification.runtime.shell is True
    assert specification.runtime.browser is True
    assert specification.runtime.code is True
    assert [requirement.model_dump() for requirement in specification.plugins] == [
        {
            "id": "local/example-plugin",
            "minimum_version": None,
            "required": True,
        },
        {
            "id": "local/optional-plugin",
            "minimum_version": "2.4.0",
            "required": False,
        },
    ]


def test_explicit_portability_declarations_override_device_defaults(tmp_path: Path):
    skill = make_skill(tmp_path)
    (skill / "SKILL.md").write_text(
        """---
name: my-skill
description: Test.
platforms: [linux, windows]
architectures: []
---

# Test
""",
        encoding="utf-8",
    )

    specification = infer_authoring_system_specification(
        skill,
        hermes_version="1.2.3",
        system_name="Darwin",
        machine="arm64",
    )

    assert specification.platforms == ["Linux", "Windows"]
    assert specification.architectures == []


def test_existing_manifest_is_preserved_instead_of_reinferred(tmp_path: Path):
    skill = make_skill(tmp_path)
    manifest = {
        "schema_version": 1,
        "name": "my-skill",
        "requirements": {
            "hermes": {"minimum_version": "0.3.0"},
            "platforms": [],
            "architectures": [],
            "model": {"capabilities": [], "minimum_context_window": None},
            "tools": [],
            "plugins": [],
            "credentials": [],
            "connections": [],
            "filesystem": {"read": [], "write": []},
            "network": {"destinations": []},
            "runtime": {
                "shell": False,
                "browser": False,
                "code": False,
                "sandbox": True,
            },
            "hardware": [],
            "known_limitations": [],
        },
    }
    original = json.dumps(manifest, separators=(",", ":")).encode()
    (skill / "skill.manifest.json").write_bytes(original)

    package = prepare_package(
        skill,
        overlay_root=tmp_path / "overlays",
        author_description="A valid description.",
        owner="owner",
        installation_id="installation-123456",
    )

    assert (package.overlay / "skill.manifest.json").read_bytes() == original


@pytest.mark.parametrize("extension", [".txt", ".md", ".rst", ".adoc", ".asciidoc"])
def test_allowlisted_inert_text_references_are_accepted(tmp_path: Path, extension: str):
    skill = make_skill(tmp_path)
    (skill / "assets" / f"guide{extension}").parent.mkdir(exist_ok=True)
    (skill / "assets" / f"guide{extension}").write_text(
        "Static reference material.\n", encoding="utf-8"
    )
    package = prepare_package(
        skill,
        overlay_root=tmp_path / "overlays",
        author_description="A valid description.",
        owner="owner",
        installation_id="installation-123456",
    )
    assert f"assets/guide{extension}" in {item.path for item in package.files}


def test_markdown_link_to_active_script_is_rejected(tmp_path: Path):
    skill = make_skill(tmp_path)
    skill_md = skill / "SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8")
        + "\nRun [scripts/run.sh](scripts/run.sh) to continue.\n",
        encoding="utf-8",
    )

    with pytest.raises(PackagePolicyError, match="active scripts/templates"):
        prepare_package(
            skill,
            overlay_root=tmp_path / "overlays",
            author_description="A valid description.",
            owner="owner",
            installation_id="installation-123456",
        )


@pytest.mark.parametrize(
    "relative,content",
    [
        ("scripts/run.sh", "echo nope"),
        ("templates/active.md", "active"),
        ("package.json", "{}"),
        ("refs/package.json", "{}"),
        ("refs/run.sh", "echo nope"),
        ("refs/template.j2", "{{ active }}"),
        ("refs/config.yaml", "active: true"),
        ("refs/.github/workflows/note.md", "inert-looking text"),
        ("assets/hooks/readme.txt", "inert-looking text"),
        ("assets/page.html", "<script>active</script>"),
        ("assets/logo.svg", "<svg></svg>"),
        ("assets/archive.zip", "not really an archive"),
        ("refs/README", "unknown extension"),
    ],
)
def test_unsupported_content_is_rejected_not_silently_omitted(
    tmp_path: Path, relative: str, content: str
):
    skill = make_skill(tmp_path)
    target = skill / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    with pytest.raises(PackagePolicyError):
        prepare_package(
            skill,
            overlay_root=tmp_path / "overlays",
            author_description="A valid description.",
            owner="owner",
            installation_id="installation-123456",
        )


def test_download_rejects_hostile_paths_modes_and_binary():
    manifest = b'{"schema_version":1}'
    base = [("SKILL.md", "file", b"# test"), ("skill.manifest.json", "file", manifest)]
    with pytest.raises(PackagePolicyError):
        verify_content_files(base + [("../escape", "file", b"x")])
    with pytest.raises(PackagePolicyError):
        verify_content_files([
            (name, "exec" if name == "SKILL.md" else mode, body)
            for name, mode, body in base
        ])
    with pytest.raises(PackagePolicyError):
        verify_content_files(base + [("assets/image.bin", "file", b"\xff\xfe")])
    with pytest.raises(PackagePolicyError, match="NUL"):
        verify_content_files(
            base + [("refs/padded.txt", "file", b"a" * 1024 + b"\x00")]
        )
    with pytest.raises(PackagePolicyError, match="shebang"):
        verify_content_files(
            base + [("refs/run.txt", "file", b"#!/bin/sh\necho unsafe")]
        )


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.txt",
        "refs//empty.txt",
        "refs/./dot.txt",
        "refs/../escape.txt",
        "refs/%2e%2e/escape.txt",
        "refs/%252e%252e/escape.txt",
        "refs/%2fescape.txt",
        "refs\\escape.txt",
        "refs/control\x00.txt",
        "refs/control\x1f.txt",
        "refs/control\x7f.txt",
        "refs/trailing.txt.",
        "refs/trailing.txt ",
        "refs/NUL.txt",
        "refs/com1.notes.txt",
        "refs/bad:name.txt",
        "refs/cafe\u0301.txt",
        "refs/a/b/c/d.txt",
    ],
)
def test_download_rejects_nonportable_or_noncanonical_paths(path: str):
    base = [
        ("SKILL.md", "file", b"# test"),
        ("skill.manifest.json", "file", b'{"schema_version":1}'),
    ]
    with pytest.raises(PackagePolicyError):
        verify_content_files(base + [(path, "file", b"reference")])


@pytest.mark.parametrize(
    "path",
    [
        "refs/run.sh",
        "assets/template.jinja",
        "refs/config.toml",
        "assets/page.html",
        "assets/logo.svg",
        "assets/image.png",
        "refs/archive.tar",
        "refs/unknown",
        "refs/SKILL.md",
        "assets/skill.manifest.json",
    ],
)
def test_download_rejects_every_file_outside_the_text_allowlist(path: str):
    base = [
        ("SKILL.md", "file", b"# test"),
        ("skill.manifest.json", "file", b'{"schema_version":1}'),
    ]
    with pytest.raises(PackagePolicyError):
        verify_content_files(base + [(path, "file", b"apparently harmless")])


def test_download_rejects_install_target_collisions():
    base = [
        ("SKILL.md", "file", b"# test"),
        ("skill.manifest.json", "file", b'{"schema_version":1}'),
    ]
    with pytest.raises(PackagePolicyError, match="collision"):
        verify_content_files(
            base
            + [
                ("refs/Notes.txt", "file", b"one"),
                ("refs/notes.txt", "file", b"two"),
            ]
        )


def test_download_enforces_file_count_per_file_and_total_caps(monkeypatch):
    base = [
        ("SKILL.md", "file", b"# test"),
        ("skill.manifest.json", "file", b'{"schema_version":1}'),
    ]
    too_many = base + [
        (f"refs/{number}.txt", "file", b"x")
        for number in range(package_module.MAX_FILES - len(base) + 1)
    ]
    with pytest.raises(PackagePolicyError, match="exceeds 32 files"):
        verify_content_files(too_many)

    with pytest.raises(PackagePolicyError, match="file exceeds"):
        verify_content_files(
            base
            + [
                (
                    "refs/large.txt",
                    "file",
                    b"x" * (package_module.MAX_FILE_BYTES + 1),
                )
            ]
        )

    monkeypatch.setattr(package_module, "MAX_TREE_BYTES", 32)
    with pytest.raises(PackagePolicyError, match="total bytes"):
        verify_content_files(base + [("refs/total.txt", "file", b"x" * 16)])


def test_local_preparation_enforces_file_size_cap(tmp_path: Path, monkeypatch):
    skill = make_skill(tmp_path)
    monkeypatch.setattr(package_module, "MAX_FILE_BYTES", 64)
    (skill / "refs" / "large.txt").write_bytes(b"x" * 65)
    with pytest.raises(PackagePolicyError, match="file exceeds"):
        prepare_package(
            skill,
            overlay_root=tmp_path / "overlays",
            author_description="A valid description.",
            owner="owner",
            installation_id="installation-123456",
        )


def test_local_preparation_rejects_duplicate_manifest_keys(tmp_path: Path):
    skill = make_skill(tmp_path)
    manifest = (
        b'{"schema_version":1,"schema_version":1,"name":"duplicate","requirements":{}}'
    )
    (skill / "skill.manifest.json").write_bytes(manifest)
    with pytest.raises(PackagePolicyError, match="manifest is invalid"):
        prepare_package(
            skill,
            overlay_root=tmp_path / "overlays",
            author_description="A valid description.",
            owner="owner",
            installation_id="installation-123456",
        )


def test_local_preparation_rejects_hard_links(tmp_path: Path):
    skill = make_skill(tmp_path)
    os.link(skill / "refs" / "notes.txt", skill / "refs" / "alias.txt")
    with pytest.raises(PackagePolicyError, match="hard-linked"):
        prepare_package(
            skill,
            overlay_root=tmp_path / "overlays",
            author_description="A valid description.",
            owner="owner",
            installation_id="installation-123456",
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no FIFO support")
def test_local_preparation_rejects_special_files(tmp_path: Path):
    skill = make_skill(tmp_path)
    os.mkfifo(skill / "refs" / "special.txt")
    with pytest.raises(PackagePolicyError, match="special filesystem"):
        prepare_package(
            skill,
            overlay_root=tmp_path / "overlays",
            author_description="A valid description.",
            owner="owner",
            installation_id="installation-123456",
        )


def test_referenced_script_requires_explicit_instruction_only_fork(tmp_path: Path):
    skill = make_skill(tmp_path)
    (skill / "SKILL.md").write_text("Run scripts/deploy.sh now.", encoding="utf-8")
    with pytest.raises(PackagePolicyError, match="instruction-only fork"):
        prepare_package(
            skill,
            overlay_root=tmp_path / "overlays",
            author_description="A valid description.",
            owner="owner",
            installation_id="installation-123456",
        )
