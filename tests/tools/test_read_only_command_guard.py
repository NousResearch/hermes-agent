"""Unit + red-team tests for tools.read_only_command_guard.

Covers: the minimum allowed command families (P0 spec), the full A-Z
red-team attack list (Rob read-only operator P0/P1 implementation task),
and a handful of additional adversarial cases the guard's own docstring
claims to handle (nested substitution, nested nulls).

The consolidated security-closure pass REMOVED validator families no
registered rob_* tool can emit (curl, find, file, rg, openssl, ip,
tailscale, dig) as dead surface — see TestDeadSurfaceRemoved.
"""

from tools.read_only_command_guard import run_read_only_guard


def allowed(cmd: str) -> bool:
    return run_read_only_guard(cmd).allowed


def denied(cmd: str) -> bool:
    return not run_read_only_guard(cmd).allowed


# ---------------------------------------------------------------------------
# Minimum allowed command families
# ---------------------------------------------------------------------------

class TestFilesystemAllowed:
    def test_ls(self):
        assert allowed("ls -la /var/log")

    def test_cat(self):
        assert allowed("cat /etc/hostname")

    def test_head_tail(self):
        assert allowed("head -n 50 file.log")
        assert allowed("tail -f /dev/null")  # -f is fine, just a flag; no redirect present

    def test_grep(self):
        assert allowed("grep -n foo file.txt")

    def test_stat_readlink(self):
        assert allowed("stat /etc/hostname")
        assert allowed("readlink -f /etc/resolv.conf")

    def test_du_df_pwd(self):
        assert allowed("du -sh /var/log")
        assert allowed("df -h /")
        assert allowed("pwd")

    def test_cd_chained_with_readonly_command(self):
        # Needed by git_inspect/docker_compose_ps's own command templates
        # (`cd <repo> && git status`) — cd itself never mutates anything,
        # its effect is scoped to the single subprocess it runs in.
        assert allowed("cd /some/repo && git status")
        assert denied("cd /some/repo && rm -rf .")


class TestProcessHostAllowed:
    def test_ps_pgrep_pstree(self):
        assert allowed("ps aux")
        assert allowed("pgrep -f hermes")
        assert allowed("pstree -p")

    def test_uptime_uname_free(self):
        assert allowed("uptime")
        assert allowed("uname -a")
        assert allowed("free -h")

    def test_id_whoami_which_whereis(self):
        assert allowed("id")
        assert allowed("whoami")
        assert allowed("which python3")
        assert allowed("whereis git")


class TestNetworkAllowed:
    def test_ss(self):
        assert allowed("ss -tlnp")

    def test_getent_nslookup(self):
        assert allowed("getent hosts example.com")
        assert allowed("nslookup example.com")


class TestDeadSurfaceRemoved:
    """Consolidated security pass: validator families with no registered
    rob_* caller were REMOVED rather than hardened for hypothetical future
    use (YAGNI). They are now denied as unrecognized executables — the
    same allowlist-first denial as any other unknown command."""

    def test_curl_removed(self):
        # http_probe goes through urllib, never a curl shellout.
        assert denied("curl -sf https://example.com/api/health")
        assert denied("curl -I https://example.com")
        assert denied("curl -X GET https://example.com")
        assert denied("curl -sf https://user:hunter2@example.com/api")

    def test_find_removed(self):
        assert denied("find /var/log -name '*.log' -mtime -1")
        assert denied("find /tmp -maxdepth 0 -fprint /tmp/x")

    def test_file_removed(self):
        assert denied("file /bin/ls")
        assert denied("file -i /bin/ls")
        assert denied("file --compile -m /tmp/x")

    def test_rg_removed(self):
        assert denied("rg --json foo .")
        assert denied("rg --pre /usr/bin/id -e . /etc/hostname")

    def test_openssl_removed(self):
        # tls_inspect uses Python ssl, never an openssl shellout.
        assert denied("openssl s_client -connect example.com:443 -quiet")

    def test_ip_removed(self):
        assert denied("ip addr")
        assert denied("ip route")

    def test_tailscale_removed(self):
        assert denied("tailscale status")
        assert denied("tailscale serve status")
        assert denied("tailscale funnel status")

    def test_dig_removed(self):
        # dig's `-f <file>` reads a file of query names and sends them to a
        # resolver — a file-read/exfiltration primitive no registered
        # tool needs; nslookup/getent (no such flag) remain allowed.
        assert denied("dig example.com")


class TestGitReadOnlyAllowed:
    def test_status_log_diff_show(self):
        assert allowed("git status")
        assert allowed("git log -5")
        assert allowed("git diff HEAD~1")
        assert allowed("git show HEAD")

    def test_branch_show_current_only(self):
        assert allowed("git branch --show-current")
        assert denied("git branch -d feature")
        assert denied("git branch -a")

    def test_rev_parse_merge_base(self):
        assert allowed("git rev-parse HEAD")
        assert allowed("git merge-base main HEAD")

    def test_worktree_list_only(self):
        assert allowed("git worktree list")
        assert denied("git worktree add /tmp/x HEAD")
        assert denied("git worktree remove /tmp/x")

    def test_tag_reflog(self):
        assert allowed("git tag")
        assert allowed("git reflog")
        assert allowed("git reflog -20")
        assert allowed("git reflog show")

    def test_tag_create_or_delete_denied(self):
        # Regression: an earlier version allowlisted the whole `tag`
        # subcommand, which let `git tag <name>` CREATE a tag and
        # `git tag -d <name>` DELETE one — real mutations, not reads.
        assert denied("git tag newtag")
        assert denied("git tag -d oldtag")
        assert denied("git tag -a v1.0 -m 'release'")

    def test_reflog_expire_denied(self):
        # Regression: `git reflog expire --all --expire=now` destroys
        # reflog history and was previously allowed by the blanket
        # `reflog` subcommand allowlist entry.
        assert denied("git reflog expire --all --expire=now")
        assert denied("git reflog delete HEAD@{0}")

    def test_diff_log_show_output_flag_denied(self):
        # Regression: `--output`/`-o` redirect a diff/log/show's output to
        # an arbitrary file — a write primitive that completely bypassed
        # the guard's separate "no redirection" rule, since it's a git
        # option rather than shell-level `>`.
        assert denied("git diff --output=/tmp/pwned")
        assert denied("git diff -o/tmp/pwned")
        assert denied("git log --output=/tmp/pwned")
        assert denied("git show --output=/tmp/pwned HEAD")


class TestDockerReadOnlyAllowed:
    def test_ps_inspect_logs(self):
        assert allowed("docker ps")
        assert allowed("docker inspect project-os-mcp")
        assert allowed("docker logs project-os-web")

    def test_stats_requires_no_stream(self):
        assert allowed("docker stats --no-stream")
        assert denied("docker stats")  # live stream is not a bounded read

    def test_network_volume_inspect(self):
        assert allowed("docker network inspect project-os-network")
        assert allowed("docker volume inspect project-os-db-data")

    def test_image_inspect_removed(self):
        # No registered tool emits `docker image inspect` — removed with
        # the other unregistered subcommand surface.
        assert denied("docker image inspect project-os-web")

    def test_compose_ps(self):
        assert allowed("docker compose ps")
        assert denied("docker compose up")
        assert denied("docker compose down")
        assert denied("docker compose restart")

    def test_compose_config_removed(self):
        assert denied("docker compose config")


class TestSystemdReadOnlyAllowed:
    def test_status_show(self):
        assert allowed("systemctl status hermes-gateway")
        assert allowed("systemctl show hermes-gateway")
        assert allowed("systemctl show hermes-gateway --property=Environment")

    def test_cat_and_list_verbs_removed(self):
        # `cat`/`list-units`/`list-timers` had no registered caller and
        # were removed with the dead surface pass.
        assert denied("systemctl cat hermes-gateway")
        assert denied("systemctl list-units")
        assert denied("systemctl list-timers")

    def test_journalctl(self):
        assert allowed("journalctl -u hermes-gateway --since '1 hour ago'")

    def test_journalctl_vacuum_denied(self):
        assert denied("journalctl --vacuum-size=100M")
        assert denied("journalctl --rotate")

    def test_journalctl_abbreviation_denied(self):
        # Regression: journalctl's own getopt_long parser accepts any
        # unambiguous prefix of a long option (confirmed against the real
        # binary — `journalctl --vacuum-tim=1s` resolves cleanly to
        # `--vacuum-time=1s`), so an exact-string denylist let every
        # abbreviation of a denied flag through even though journalctl
        # itself still executed it as the full flag. The fix switched this
        # validator to an allowlist, which has no such gap: an
        # abbreviation of an unlisted flag still isn't an exact match for
        # anything in the allowed set.
        assert denied("journalctl --rotat")
        assert denied("journalctl --vacuum-tim=1s")
        assert denied("journalctl --vacuum-s=1M")
        assert denied("journalctl --flus")
        assert denied("journalctl --sy")
        assert denied("journalctl --relinquish-va")

    def test_journalctl_legitimate_forms_still_allowed(self):
        assert allowed("journalctl --no-pager -u hermes-gateway.service")
        assert allowed("journalctl -p err")
        assert allowed("journalctl -u hermes-gateway --until '5 minutes ago'")

    def test_journalctl_attached_and_equals_value_forms_allowed(self):
        assert allowed("journalctl -uhermes-gateway.service --since=now")
        assert allowed("journalctl --priority=err")

    def test_journalctl_case_variant_denied(self):
        assert denied("journalctl --NO-PAGER")
        assert denied("journalctl --UNIT=hermes-gateway")

    def test_journalctl_pipeline_into_allowed_grep(self):
        # rob_journal_query appends `| grep <pattern>` — the guard
        # validates the WHOLE pipeline, including the grep segment.
        assert allowed("journalctl --no-pager | grep error")
        assert denied("journalctl --no-pager | tee /tmp/x")


class TestCommandGuardResidual:
    """Remaining command-guard cases from the closure pass threat model:
    case variation, absolute executable paths, env wrappers, and `--`
    handling on the families that survive."""

    def test_case_variant_exe_denied(self):
        assert denied("GIT status")
        assert denied("Docker ps")

    def test_absolute_executable_path_denied(self):
        assert denied("/bin/ls /tmp")
        assert denied("/usr/bin/git status")

    def test_env_wrapper_denied(self):
        assert denied("env PATH=/usr/bin git status")
        assert denied("HOME=/etc git status")

    def test_git_double_dash_handled(self):
        assert allowed("git log --")

    def test_docker_case_variant_subcommand_denied(self):
        assert denied("docker PS")
        assert denied("docker Inspect x")


# ---------------------------------------------------------------------------
# Hard-deny families (explicit spec list)
# ---------------------------------------------------------------------------

class TestHardDenyExplicit:
    def test_sudo_su(self):
        assert denied("sudo ls")
        assert denied("su root")

    def test_destructive_fs_verbs(self):
        for cmd in ["rm -rf /tmp/x", "mv a b", "cp a b", "touch x",
                    "mkdir x", "truncate -s 0 x", "chmod 777 x",
                    "chown root x", "ln -s a b", "tee out.txt"]:
            assert denied(cmd), cmd

    def test_inplace_editors(self):
        assert denied("sed -i s/a/b/ file")
        assert denied("perl -pi -e 's/a/b/' file")

    def test_awk_system(self):
        assert denied("awk 'BEGIN{system(\"ls\")}'")

    def test_xargs_to_mutator(self):
        assert denied("find . -name '*.tmp' | xargs rm")

    def test_find_exec(self):
        assert denied("find . -name '*.tmp' -exec rm {} +")

    def test_interpreter_one_liners(self):
        assert denied("python -c 'import os; os.system(\"rm -rf /\")'")
        assert denied("node -e \"require('fs').unlinkSync('/x')\"")

    def test_shell_wrappers(self):
        assert denied("sh -c 'rm -rf /'")
        assert denied("bash -c 'rm -rf /'")

    def test_git_mutations(self):
        for cmd in ["git add .", "git commit -m x", "git push", "git pull",
                    "git fetch", "git checkout main", "git switch main",
                    "git reset --hard", "git clean -fd", "git rebase main",
                    "git merge main"]:
            assert denied(cmd), cmd

    def test_docker_mutations(self):
        for cmd in ["docker start x", "docker stop x", "docker restart x",
                    "docker rm x", "docker compose up", "docker compose down",
                    "docker compose restart", "docker exec x rm /y"]:
            assert denied(cmd), cmd

    def test_systemctl_mutations(self):
        for cmd in ["systemctl start x", "systemctl stop x",
                    "systemctl restart x", "systemctl enable x",
                    "systemctl disable x"]:
            assert denied(cmd), cmd

    def test_kill_family(self):
        assert denied("kill -9 1234")
        assert denied("pkill -f foo")

    def test_firewall(self):
        assert denied("iptables -L")  # even the "read" form denied — narrow allowlist, no exception carved
        assert denied("nft list ruleset")

    def test_tailscale_serve_mutation(self):
        assert denied("tailscale serve --bg --https=443 http://127.0.0.1:3200")
        assert denied("tailscale up")

    def test_package_managers(self):
        assert denied("apt-get install foo")
        assert denied("pip install foo")

    def test_reboot_shutdown(self):
        assert denied("reboot")
        assert denied("shutdown -h now")

    def test_db_mutation_raw_psql_denied_outright(self):
        # Raw psql is denied unconditionally — DB access must go through
        # the dedicated db_select tool, never a raw client invocation.
        assert denied("psql -U projectos -c 'select 1'")

    def test_pager_and_editor_shell_escape(self):
        assert denied("less file.txt")
        assert denied("more file.txt")
        assert denied("vim file.txt")

    def test_redirects_all_forms(self):
        assert denied("cat file > target")
        assert denied("echo x >> target")
        assert denied("cat < input")
        assert denied("cat <<< 'heredoc'")

    def test_redirect_to_dev_null_is_harmless(self):
        assert allowed("which foo > /dev/null")
        assert allowed("ls 2>/dev/null")


# ---------------------------------------------------------------------------
# Red-team A-Z (task's own attack list)
# ---------------------------------------------------------------------------

class TestRedTeamAZ:
    def test_A_redirect(self):
        assert denied("cat file > target")

    def test_B_append(self):
        assert denied("echo x >> target")

    def test_C_subshell(self):
        assert denied("$(touch /tmp/x)")

    def test_D_backticks(self):
        assert denied("`touch /tmp/x`")

    def test_E_pipeline_into_mutator(self):
        assert denied("cat file | tee /tmp/x")
        assert denied("ls | xargs rm")

    def test_F_xargs_mutation(self):
        assert denied("echo file | xargs rm")

    def test_G_find_exec_mutation(self):
        # find itself was removed from the allowlist in the dead-surface
        # pass, so this is denied even before reaching -exec.
        assert denied("find . -exec rm {} \\;")

    def test_H_python_c(self):
        assert denied("python -c 'print(1)'")

    def test_I_node_e(self):
        assert denied("node -e 'console.log(1)'")

    def test_J_bash_c(self):
        assert denied("bash -c 'echo hi'")

    def test_K_git_add(self):
        assert denied("git add .")

    def test_L_git_checkout(self):
        assert denied("git checkout main")

    def test_M_git_fetch(self):
        assert denied("git fetch origin")

    def test_N_docker_restart(self):
        assert denied("docker restart project-os-web")

    def test_O_docker_exec_rm(self):
        assert denied("docker exec project-os-mcp rm -rf /tmp")

    def test_P_systemctl_restart(self):
        assert denied("systemctl restart hermes-gateway")

    def test_Q_curl_post(self):
        # curl was removed from the allowlist in the dead-surface pass, so
        # any curl invocation — GET or POST — is denied as unrecognized.
        assert denied("curl -X POST https://example.com/api")
        assert denied("curl -d 'x=1' https://example.com/api")

    def test_S_read_env_file(self):
        # `cat` itself is an allowed read-only verb in general, but the
        # sensitive-path integration (tools/sensitive_path_guard.py,
        # reusing agent/file_safety.py) denies THIS specific target —
        # see tests/tools/test_sensitive_path_guard.py for the full
        # positive/negative matrix. Confirming the composed result here.
        assert denied("cat .env")

    def test_T_read_ssh_key(self):
        # Not covered by agent/file_safety.py at all (verified directly:
        # it returns None for SSH keys) — covered by
        # sensitive_path_guard.py's own explicit private-key-material
        # check instead. See test_sensitive_path_guard.py.
        assert denied("cat ~/.ssh/id_ed25519")

    def test_U_docker_inspect_embedded_password(self):
        # `docker inspect` takes a container name, not a filesystem path —
        # out of sensitive_path_guard's scope by design. The guard allows
        # the command; secret_redaction.py is responsible for scrubbing any
        # embedded DATABASE_URL password in the *output*.
        assert allowed("docker inspect project-os-mcp")

    def test_V_sql_insert(self):
        assert denied("psql -c 'INSERT INTO x VALUES (1)'")

    def test_W_sql_dangerous_function_via_raw_psql(self):
        assert denied("psql -c \"select pg_read_file('/etc/passwd')\"")

    def test_X_pager_shell_escape(self):
        assert denied("less file.txt")

    def test_Y_symlink_traversal(self):
        # Guard-level: `readlink`/`cat` themselves are allowed verbs (the
        # traversal-safety property belongs to agent/file_safety.py +
        # sensitive_path_guard.py, same boundary as S/T/U above) — confirm
        # the guard doesn't accidentally block ordinary symlink inspection.
        assert allowed("readlink -f /etc/some-symlink")

    def test_Z_remote_host_profile_write_attempt(self):
        # Host-profile routing (host_profiles.py) must run every command
        # through this SAME guard regardless of target host — a write
        # attempt is denied identically whether local or remote, since the
        # command string itself is what's classified, not the transport.
        assert denied("rm -rf /tmp/x")  # verified again here as the guard's own contribution to Z


# ---------------------------------------------------------------------------
# Nested / obfuscated adversarial cases beyond the explicit list
# ---------------------------------------------------------------------------

class TestNestedObfuscation:
    def test_nested_substitution_mutator(self):
        assert denied("echo $(rm -rf /tmp/x)")

    def test_quoted_command_word_still_detected(self):
        assert denied("'r'm -rf /tmp/x")

    def test_semicolon_chain_any_segment_denied(self):
        assert denied("ls; rm -rf /tmp/x")

    def test_and_or_chain_any_segment_denied(self):
        assert denied("ls && rm -rf /tmp/x")
        assert denied("ls || rm -rf /tmp/x")

    def test_background_operator_segment_denied(self):
        assert denied("ls & rm -rf /tmp/x")

    def test_brace_group_mutator_denied(self):
        assert denied("{ rm -rf /tmp/x; }")

    def test_all_segments_must_be_allowed(self):
        assert allowed("git status; ls; docker ps")
        assert denied("git status; ls; docker restart x")
