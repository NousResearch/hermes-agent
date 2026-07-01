"""Regression tests for terminal stdout decoding on Windows code pages."""

from tools.environments.base import _decode_process_output


def test_decode_process_output_preserves_utf8_russian_with_invalid_tail():
    data = "Проверка".encode("utf-8") + b"\xff"

    decoded = _decode_process_output(data)

    assert decoded == "Проверка�"


def test_decode_process_output_recovers_cp866_windows_console_russian():
    data = "Успешно: процесс завершен.".encode("cp866")

    decoded = _decode_process_output(data)

    assert decoded == "Успешно: процесс завершен."
    assert "�" not in decoded


def test_decode_process_output_keeps_utf8_as_first_choice():
    decoded = _decode_process_output("Привет из bash".encode("utf-8"))

    assert decoded == "Привет из bash"
