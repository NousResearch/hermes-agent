"""Round-trip YAML writes must not mutate stored values (#119844)."""

from utils import _roundtrip_dump, _roundtrip_load


def _long_backslash_value() -> str:
    # A fold point right after an escaped backslash is what corrupted stored config values.
    return "A" * 74 + "D:\\CentBrowserPortable " + "B" * 40


def test_long_double_quoted_backslash_scalar_survives_save(tmp_path):
    value = _long_backslash_value()
    path = tmp_path / "config.yaml"
    path.write_text('k: "' + value.replace("\\", "\\\\") + '"\n', encoding="utf-8")

    yaml_rt, data = _roundtrip_load(path)
    assert data["k"] == value

    _roundtrip_dump(path, yaml_rt, data)
    _, reloaded = _roundtrip_load(path)
    assert reloaded["k"] == value


def test_noop_save_is_a_fixed_point(tmp_path):
    value = _long_backslash_value()
    path = tmp_path / "config.yaml"
    path.write_text('k: "' + value.replace("\\", "\\\\") + '"\n', encoding="utf-8")

    for _ in range(3):
        yaml_rt, data = _roundtrip_load(path)
        assert data["k"] == value
        _roundtrip_dump(path, yaml_rt, data)

    _, reloaded = _roundtrip_load(path)
    assert reloaded["k"] == value
