from __future__ import annotations

import pytest

from cron.jobs import save_jobs, use_cron_store

import tools.skill_runtime_contracts as runtime_contracts
from tools.skill_runtime_contracts import (
    SkillRuntimeScanError,
    blocking_skill_runtime_references,
    find_skill_runtime_references,
)


def test_finds_wrapper_skill_reference(tmp_path):
    hermes_home = tmp_path / ".hermes"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "run_daily.sh"
    wrapper.write_text("hermes chat -s legacy-skill 'run brief'\n", encoding="utf-8")

    refs = find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert len(refs) == 1
    assert refs[0].surface == "hermes.bin"
    assert refs[0].path == str(wrapper)
    assert refs[0].line == 1


def test_does_not_match_skill_name_as_substring(tmp_path):
    hermes_home = tmp_path / ".hermes"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "run_daily.sh").write_text(
        "hermes chat -s legacy-skill-v2 'run brief'\n",
        encoding="utf-8",
    )

    assert find_skill_runtime_references("legacy-skill", hermes_home=hermes_home) == []


def test_finds_supported_shell_invocation_forms(tmp_path):
    hermes_home = tmp_path / ".hermes"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "run_daily.sh"
    wrapper.write_text(
        "env MODE=batch /usr/local/bin/hermes chat --skills other,legacy-skill\n"
        "command hermes chat --skills=legacy-skill\n"
        "python3 -m hermes_cli.main chat -slegacy-skill\n",
        encoding="utf-8",
    )

    refs = find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert [ref.line for ref in refs] == [1, 2, 3]


def test_matches_only_path_identifiers_that_resolve_to_target_skill(tmp_path):
    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    skill_dir = skills_root / "operations" / "legacy-skill"
    skill_dir.mkdir(parents=True)
    outside_skill = tmp_path / "outside" / "legacy-skill"
    outside_skill.mkdir(parents=True)

    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "run_daily.sh"
    wrapper.write_text(
        "hermes chat -s operations/legacy-skill\n"
        f"hermes chat -s {skill_dir}\n"
        "hermes chat --skills=other,operations/legacy-skill\n"
        'hermes chat -s "operations\\legacy-skill"\n'
        f"hermes chat -s {outside_skill}\n"
        "hermes chat -s ../outside/legacy-skill\n",
        encoding="utf-8",
    )

    refs = find_skill_runtime_references(
        "legacy-skill",
        hermes_home=hermes_home,
        skill_path=skill_dir,
        skills_root=skills_root,
    )

    assert [ref.line for ref in refs] == [1, 2, 3, 4]


def test_finds_shell_control_flow_functions_and_launch_wrappers(tmp_path):
    hermes_home = tmp_path / ".hermes"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "run_daily.sh"
    wrapper.write_text(
        "if ready; then hermes chat -s legacy-skill; fi\n"
        "run_daily() { hermes chat -s legacy-skill; }\n"
        "sudo -u root hermes chat -s legacy-skill\n"
        "nohup hermes chat -s legacy-skill >/tmp/hermes.log 2>&1 &\n"
        "env -i MODE=batch command hermes chat -s legacy-skill\n"
        "time -p hermes chat -s legacy-skill\n"
        "while ready; do exec hermes chat -s legacy-skill; done\n",
        encoding="utf-8",
    )

    refs = find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert [ref.line for ref in refs] == [1, 2, 3, 4, 5, 6, 7]


def test_runtime_reference_text_does_not_expose_command_secrets(tmp_path):
    hermes_home = tmp_path / ".hermes"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    secret = "do-not-return-this-token"
    (bin_dir / "run_daily.sh").write_text(
        f"API_TOKEN={secret} hermes chat -s legacy-skill\n",
        encoding="utf-8",
    )

    refs = find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert len(refs) == 1
    assert secret not in refs[0].text
    assert "legacy-skill" in refs[0].text


def test_finds_explicit_python_process_invocations(tmp_path):
    hermes_home = tmp_path / ".hermes"
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "run_daily.task"
    script.write_text(
        "import os\n"
        "import subprocess\n"
        "subprocess.run(['/usr/local/bin/hermes', 'chat', '--skills', 'other,legacy-skill'])\n"
        "os.system('hermes chat -s legacy-skill')\n",
        encoding="utf-8",
    )

    refs = find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert [ref.line for ref in refs] == [3, 4]
    assert all(ref.surface == "hermes.scripts" for ref in refs)


def test_ignores_comments_prompts_and_data_that_only_mention_skill(tmp_path):
    hermes_home = tmp_path / ".hermes"
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run_daily.sh").write_text(
        "# hermes chat -s legacy-skill\n"
        "PROMPT='Tell the operator to run hermes chat -s legacy-skill'\n"
        "MULTILINE_PROMPT='\n"
        "hermes chat -s legacy-skill\n"
        "'\n"
        "printf '%s\\n' 'legacy-skill'\n"
        "cat <<'PROMPT'\n"
        "hermes chat -s legacy-skill\n"
        "PROMPT\n"
        "hermes chat -s legacy-skill-v2\n",
        encoding="utf-8",
    )
    (scripts_dir / "prompt_data.json").write_text(
        '{"prompt": "hermes chat -s legacy-skill"}\n',
        encoding="utf-8",
    )
    (scripts_dir / "notes.py").write_text(
        '"""Example: hermes chat -s legacy-skill."""\n'
        "example = 'hermes chat -s legacy-skill'\n",
        encoding="utf-8",
    )

    assert find_skill_runtime_references("legacy-skill", hermes_home=hermes_home) == []


def test_cron_references_are_blocking(tmp_path):
    hermes_home = tmp_path / ".hermes"
    with use_cron_store(hermes_home):
        save_jobs([
            {
                "id": "job-1",
                "name": "daily",
                "skills": ["legacy-skill"],
                "skill": "legacy-skill",
            }
        ])

    refs = find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert len(refs) == 1
    assert refs[0].surface == "cron.jobs"
    assert (
        blocking_skill_runtime_references("legacy-skill", hermes_home=hermes_home)
        == refs
    )


def test_cron_path_identifiers_must_resolve_to_target_skill(tmp_path):
    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    skill_dir = skills_root / "operations" / "legacy-skill"
    skill_dir.mkdir(parents=True)
    outside_skill = tmp_path / "outside" / "legacy-skill"
    outside_skill.mkdir(parents=True)

    with use_cron_store(hermes_home):
        save_jobs([
            {"id": "relative", "skills": ["operations/legacy-skill"]},
            {"id": "absolute", "skills": [str(skill_dir)]},
            {"id": "windows", "skills": [r"operations\legacy-skill"]},
            {"id": "outside", "skills": [str(outside_skill)]},
            {"id": "traversal", "skills": ["../outside/legacy-skill"]},
        ])

    refs = find_skill_runtime_references(
        "legacy-skill",
        hermes_home=hermes_home,
        skill_path=skill_dir,
        skills_root=skills_root,
    )

    assert [ref.path.rsplit("#", 1)[-1] for ref in refs] == [
        "relative",
        "absolute",
        "windows",
    ]


def test_corrupt_cron_store_fails_closed(tmp_path):
    hermes_home = tmp_path / ".hermes"
    cron_dir = hermes_home / "cron"
    cron_dir.mkdir(parents=True)
    jobs_file = cron_dir / "jobs.json"
    jobs_file.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SkillRuntimeScanError) as raised:
        find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert raised.value.surface == "cron.jobs"
    assert raised.value.path == str(jobs_file)


def test_unreadable_runtime_contents_fail_closed(tmp_path):
    hermes_home = tmp_path / ".hermes"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "run_daily.sh"
    wrapper.write_bytes(b"\xff")

    with pytest.raises(SkillRuntimeScanError) as raised:
        find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert raised.value.surface == "hermes.bin"
    assert raised.value.path == str(wrapper)


def test_oversized_runtime_file_fails_closed(tmp_path):
    hermes_home = tmp_path / ".hermes"
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "run_daily.sh"
    wrapper.write_bytes(b"#" * (runtime_contracts._MAX_SCAN_BYTES + 1))

    with pytest.raises(SkillRuntimeScanError) as raised:
        find_skill_runtime_references("legacy-skill", hermes_home=hermes_home)

    assert raised.value.surface == "hermes.bin"
    assert raised.value.path == str(wrapper)
