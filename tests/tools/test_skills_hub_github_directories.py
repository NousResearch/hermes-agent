"""GitHub skill bundles distinguish referenced directories from unsafe entries."""

from unittest.mock import MagicMock, patch

import pytest

from tools.skills_hub_github import GitHubAuth, GitHubSource


@pytest.mark.parametrize("skill_dir", ["skills/demo", ""])
def test_fetch_keeps_files_under_referenced_directories(skill_dir):
    """Directory mentions in SKILL.md must retain the complete bundled subtree."""
    prefix = f"{skill_dir}/" if skill_dir else ""
    skill_md = (
        "---\nname: demo\n---\n\n"
        "Bundled scripts: `scripts/lib/` and `scripts/lib/vendor/bird-search/`.\n"
        "Read [guides](references/guides/).\n"
    )
    contents = {
        "scripts/lib/env.py": b"CONFIG = {}\n",
        "scripts/lib/vendor/bird-search/index.js": b"export const search = () => [];\n",
        "references/guides/setup.md": b"# Setup\n",
    }
    directories = [
        "scripts", "scripts/lib", "scripts/lib/vendor", "scripts/lib/vendor/bird-search",
        "references", "references/guides",
    ]
    entries = [
        {"path": prefix + path, "type": "tree", "mode": "040000"}
        for path in directories
    ] + [
        {"path": prefix + path, "type": "blob", "mode": "100644"}
        for path in contents
    ]
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))

    def fetch_bytes(_repo, path, **_kwargs):
        return contents[path.removeprefix(prefix)]

    with patch.object(source, "_fetch_file_content", return_value=skill_md), \
         patch.object(source, "_get_repo_tree", return_value=("main", entries)), \
         patch.object(source, "_fetch_file_bytes", side_effect=fetch_bytes):
        bundle = source.fetch(f"owner/repo/{skill_dir}")

    assert bundle is not None
    assert bundle.files == {"SKILL.md": skill_md, **contents}


@pytest.mark.parametrize("entry_type,mode", [("blob", "120000"), ("commit", "160000")])
def test_fetch_rejects_referenced_symlink_or_submodule(entry_type, mode):
    """A directory-shaped reference must never make a linked entry acceptable."""
    skill_md = "---\nname: demo\n---\n\nUse the bundled `scripts/lib/`.\n"
    entries = [
        {"path": "skills/demo/scripts", "type": "tree", "mode": "040000"},
        {"path": "skills/demo/scripts/lib", "type": entry_type, "mode": mode},
        {"path": "skills/demo/scripts/run.py", "type": "blob", "mode": "100644"},
    ]
    source = GitHubSource(auth=MagicMock(spec=GitHubAuth))
    with patch.object(source, "_fetch_file_content", return_value=skill_md), \
         patch.object(source, "_get_repo_tree", return_value=("main", entries)), \
         patch.object(source, "_fetch_file_bytes", return_value=b"print('ready')\n"):
        bundle = source.fetch("owner/repo/skills/demo")

    assert bundle is None
