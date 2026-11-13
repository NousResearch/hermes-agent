"""Regression coverage for unnecessary execution-flag parsing work."""

from tools import approval_detection_exec as execution


def test_unrelated_command_words_do_not_require_option_tokenization(monkeypatch):
    calls = []
    original = execution._shell_segment_tokens

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_shell_segment_tokens", counted)
    assert list(execution._execution_flag_findings("echo hello")) == []
    assert calls == []
