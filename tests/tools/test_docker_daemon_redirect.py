"""Docker/Podman daemon-redirect and lifecycle flag-insertion detection.

Inspired by Claude Code 2.1.214, which added permission prompts for docker
commands (including the Podman ``docker`` shim) carrying daemon-redirect
flags (``--url``, ``--connection``, ``--identity``, and Podman's remote
mode) that previously ran without one.

A daemon redirect makes a local-looking command operate on a different
(often remote) daemon, so any docker/podman invocation carrying one
requires approval regardless of subcommand.
"""

from tools.approval import detect_dangerous_command


class TestDockerDaemonRedirect:
    def test_docker_dash_h_remote_host(self):
        is_dangerous, key, desc = detect_dangerous_command(
            "docker -H ssh://prod-host stop app")
        assert is_dangerous is True
        assert key is not None
        assert "daemon redirect" in desc

    def test_docker_long_host_flag_equals_form(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "docker --host=tcp://10.0.0.5:2375 ps")
        assert is_dangerous is True
        assert "daemon redirect" in desc

    def test_docker_host_flag_after_other_global_flags(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "docker --log-level debug -H tcp://10.0.0.5:2375 images")
        assert is_dangerous is True
        assert "daemon redirect" in desc

    def test_docker_context_flag(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "docker --context production rm -f db")
        assert is_dangerous is True
        assert "daemon redirect" in desc

    def test_docker_context_use(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "docker context use production")
        assert is_dangerous is True
        assert "context use" in desc

    def test_docker_host_env_prefix(self):
        is_dangerous, _, _ = detect_dangerous_command(
            "DOCKER_HOST=ssh://prod docker stop app")
        assert is_dangerous is True

    def test_docker_context_env_prefix(self):
        is_dangerous, _, _ = detect_dangerous_command(
            "DOCKER_CONTEXT=production docker ps")
        assert is_dangerous is True

    def test_container_host_env_prefix(self):
        is_dangerous, _, _ = detect_dangerous_command(
            "CONTAINER_HOST=ssh://root@prod:22/run/podman/podman.sock podman ps")
        assert is_dangerous is True

    def test_podman_url_flag(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "podman --url ssh://core@remote:22/run/podman.sock ps")
        assert is_dangerous is True
        assert "daemon redirect" in desc

    def test_podman_connection_flag(self):
        is_dangerous, _, _ = detect_dangerous_command(
            "podman --connection prod rm -f web")
        assert is_dangerous is True

    def test_podman_identity_flag(self):
        is_dangerous, _, _ = detect_dangerous_command(
            "podman --identity ~/.ssh/id_ed25519 --url ssh://x ps")
        assert is_dangerous is True

    def test_podman_remote_mode(self):
        is_dangerous, _, desc = detect_dangerous_command("podman --remote ps")
        assert is_dangerous is True
        assert "remote mode" in desc

    def test_podman_short_remote_flag(self):
        is_dangerous, _, _ = detect_dangerous_command("podman -r images")
        assert is_dangerous is True

    # -- negatives: local docker usage stays out of the deny ----------------

    def test_plain_docker_ps_not_flagged(self):
        assert detect_dangerous_command("docker ps -a") == (False, None, None)

    def test_docker_run_not_flagged(self):
        assert detect_dangerous_command(
            "docker run --rm -it alpine sh") == (False, None, None)

    def test_docker_bare_help_flag_not_flagged(self):
        # `docker -h` alone is help; the redirect rule requires a value token.
        assert detect_dangerous_command("docker -h") == (False, None, None)

    def test_docker_run_hostname_flag_not_flagged(self):
        # `-h` in the subcommand position is `docker run --hostname`.
        assert detect_dangerous_command(
            "docker run -h myhost alpine") == (False, None, None)

    def test_docker_build_not_flagged(self):
        assert detect_dangerous_command(
            "docker build -t myimage .") == (False, None, None)

    def test_docker_context_ls_not_flagged(self):
        assert detect_dangerous_command(
            "docker context ls") == (False, None, None)

    def test_podman_local_ps_not_flagged(self):
        assert detect_dangerous_command("podman ps") == (False, None, None)

    def test_podman_local_rm_not_misattributed_to_redirect(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "podman rm old-container")
        if is_dangerous:
            assert "remote" not in desc


class TestDockerLifecycleFlagInsertion:
    """Global flags must not slip a lifecycle verb past the docker guard."""

    def test_docker_stop_still_flagged(self):
        is_dangerous, _, desc = detect_dangerous_command("docker stop app")
        assert is_dangerous is True
        assert "container lifecycle" in desc

    def test_docker_stop_with_global_flag_flagged(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "docker --log-level debug stop app")
        assert is_dangerous is True
        assert "container lifecycle" in desc

    def test_docker_compose_down_with_file_flag_flagged(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "docker compose -f docker-compose.prod.yml down")
        assert is_dangerous is True
        assert "container lifecycle" in desc

    def test_legacy_docker_compose_binary_down_flagged(self):
        is_dangerous, _, desc = detect_dangerous_command("docker-compose down")
        assert is_dangerous is True
        assert "container lifecycle" in desc

    def test_docker_compose_up_not_flagged(self):
        assert detect_dangerous_command(
            "docker compose -f dev.yml up -d") == (False, None, None)

    def test_docker_run_restart_policy_not_flagged(self):
        assert detect_dangerous_command(
            "docker run --restart=always -d nginx") == (False, None, None)
