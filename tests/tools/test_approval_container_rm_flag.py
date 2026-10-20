"""The ``rm`` embedded in a container/CLI flag is not an ``rm`` command (#132313).

The DANGEROUS rm rules anchored on a bare word boundary, which also matches the ``rm``
inside ``--rm``: ``docker run --rm --network=none alpine true`` read as ``rm --network...``
and tripped the recursive-delete rules (with ``-v /host:/x`` even the root-path rule), so
``approvals test`` asked for approval and ``single_query_mode: deny`` blocked the run.
The rules now bar a leading dash; real recursive deletions still require approval.
"""

from tools.approval import detect_dangerous_command


def test_docker_rm_flag_not_flagged():
    # The `--network=none` / `-v /host:/x` / `ls -r` suffixes previously fed the `--rm`
    # fragment into the recursive-delete and root-path rules as if it were `rm <flags>`.
    for command in (
        "docker run --rm --network=none alpine:3.20 true",
        "docker run --rm alpine:3.20 true",
        "docker run --rm -v /host:/x alpine:3.20 true",
        "docker run --rm alpine:3.20 ls -r",
        "podman run --rm --network=none alpine:3.20 true",
    ):
        is_dangerous, key, desc = detect_dangerous_command(command)
        assert is_dangerous is False, f"{command!r} should be safe, got: {desc}"
        assert key is None


def test_real_recursive_rm_still_flagged_after_flag_fix():
    for command in (
        "rm -rf /tmp/x",
        "rm -r mydir",
        "rm --recursive /tmp",
        "sudo rm -rf /tmp/x",
        "foo; rm -r bar",
        "echo x && rm -rf /tmp/y",
        "rm build/ -rf",
    ):
        is_dangerous, key, desc = detect_dangerous_command(command)
        assert is_dangerous is True, f"{command!r} should require approval"
        assert "delete" in desc.lower()
