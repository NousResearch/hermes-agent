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


    def test_container_host_env_prefix(self):
        is_dangerous, _, _ = detect_dangerous_command(
            "CONTAINER_HOST=ssh://root@prod:22/run/podman/podman.sock podman ps")
        assert is_dangerous is True

    def test_podman_url_flag(self):
        is_dangerous, _, desc = detect_dangerous_command(
            "podman --url ssh://core@remote:22/run/podman.sock ps")
        assert is_dangerous is True
        assert "daemon redirect" in desc


    # -- negatives: local docker usage stays out of the deny ----------------

    def test_plain_docker_ps_not_flagged(self):
        assert detect_dangerous_command("docker ps -a") == (False, None, None)


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


    def test_docker_run_restart_policy_not_flagged(self):
        assert detect_dangerous_command(
            "docker run --restart=always -d nginx") == (False, None, None)
