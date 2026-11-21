"""Background startup warning filtering regression for #139213."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.parametrize("separate_reads", [False, True])
def test_reader_filters_zle_startup_warnings_but_keeps_command_output(
    monkeypatch, separate_reads
):
    warning = b"(eval):1: can't change option: zle\n"
    payload = b"command output\n" + warning
    chunks = [warning, warning, warning, payload] if separate_reads else [warning * 3 + payload]
    process = SimpleNamespace(
        stdout=SimpleNamespace(buffer=SimpleNamespace(read1=Mock(side_effect=[*chunks, b""]))),
        wait=lambda: 0,
        returncode=0,
    )
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_noise", command="fixture", task_id="noise-test")
    session.process = process
    monkeypatch.setattr(registry, "_move_to_finished", lambda _session: None)
    registry._reader_loop(session)
    assert session.output_buffer == payload.decode()


@pytest.mark.parametrize("text", ["", "normal output\n", "normal output\n(eval):1: can't change option: zle\n"])
def test_clean_shell_noise_preserves_nonstartup_output(text):
    assert ProcessRegistry._clean_shell_noise(text) == text
