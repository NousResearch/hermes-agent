"""The TUI raw-config cache never treats an unreadable config.yaml as an absent one.

``_load_cfg_raw`` feeds ``config.set`` read-modify-write round-trips; serving its cached ``{}`` for a
config that now exists but cannot be stat'ed would let the next save erase every other setting.
"""

from hermes_cli import config_backend
from hermes_cli.config_read_errors import FailedConfigRead
from tui_gateway import server


def test_stat_error_after_a_cached_absent_config_is_a_failed_read(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    monkeypatch.setattr(server, "_active_config_path", lambda: cfg)
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_sig", None)
    monkeypatch.setattr(server, "_cfg_path", None)
    assert server._load_cfg_raw() == {}  # absent: cached as "no config"

    cfg.write_text("display:\n  skin: mono\nmodel:\n  default: keep-me\n", encoding="utf-8")

    def denied(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(config_backend, "config_version", denied)
    assert isinstance(server._load_cfg_raw(), FailedConfigRead)
