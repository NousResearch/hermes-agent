"""`hermes config set <key> []` / `{}` must not delete the comment block that introduces the *next* key.

ruamel stores the comment lines that follow a collection's last entry (blank line, a section banner,
the next key's own description) on that last entry. Emptying or truncating the collection drops the
entry and the next section's comments with it, while the file still loads, so nothing reports it.
"""

import difflib
from pathlib import Path

import pytest

import hermes_yaml as yaml

from utils import atomic_roundtrip_yaml_save, atomic_roundtrip_yaml_update

EXAMPLE = Path(__file__).resolve().parents[1] / "cli-config.yaml.example"


@pytest.mark.parametrize(
    "removed, doc",
    [
        ("emptied list", "sec:\n  - a\n  # NOTE below\nb: 2\n"),
        ("emptied map", "a: 1\nsec:\n  x: 1\n# NOTE below\nb: 2\n"),
        ("list tail", "sec:\n  - a\n  - b\n\n# NOTE below\nb: 2\n"),
        ("whole section", "a: 1\nsec:\n  x: 1\n  w: 2\n\n# NOTE below\nb: 2\n"),
    ],
)
def test_save_keeps_the_next_keys_comment(tmp_path, removed, doc):
    path = tmp_path / "config.yaml"
    path.write_text(doc, encoding="utf-8")
    state = yaml.safe_load(doc)
    if removed == "whole section":
        state.pop("sec")
    elif removed == "list tail":
        state["sec"].pop()
    else:
        state["sec"].clear()

    atomic_roundtrip_yaml_save(path, state)

    text = path.read_text(encoding="utf-8")
    assert "# NOTE below" in text, text
    assert yaml.safe_load(text) == state


def _last_line(node) -> int:
    """0-based line of the last data entry inside a non-empty round-trip collection."""
    if isinstance(node, dict):
        key = next(reversed(node))
        child, line = node[key], node.lc.data[key][2]
    else:
        child, line = node[-1], node.lc.data[len(node) - 1][0]
    return _last_line(child) if isinstance(child, (dict, list)) and child else line


def _shipped_cases():
    data = yaml.roundtrip_yaml().load(EXAMPLE.read_text(encoding="utf-8"))
    cases = []
    for key, value in data.items():
        if isinstance(value, (dict, list)) and value:
            cases.append(pytest.param(key, _last_line(value), id=key))
            if isinstance(value, dict):
                cases += [
                    pytest.param(f"{key}.{k}", _last_line(v), id=f"{key}.{k}")
                    for k, v in value.items()
                    if isinstance(v, (dict, list)) and v
                ]
    return cases


@pytest.mark.parametrize("key_path, last", _shipped_cases())
def test_config_set_empty_keeps_every_comment_after_the_block(tmp_path, key_path, last):
    """Every collection in the shipped example config: emptying it may drop the block's own
    commented options, never a comment line that comes after the block's last data line."""
    src = EXAMPLE.read_text(encoding="utf-8")
    lines = src.splitlines()
    current = yaml.safe_load(src)
    for part in key_path.split("."):
        current = current[part]
    path = tmp_path / "config.yaml"
    path.write_text(src, encoding="utf-8")

    atomic_roundtrip_yaml_update(
        path, key_path, [] if isinstance(current, list) else {}
    )

    after = path.read_text(encoding="utf-8").splitlines()
    ops = difflib.SequenceMatcher(None, lines, after, autojunk=False).get_opcodes()
    lost = [
        lines[i]
        for tag, i1, i2, _, _ in ops
        if tag in ("delete", "replace")
        for i in range(i1, i2)
        if i > last and lines[i].lstrip().startswith("#")
    ]
    assert not lost, (
        f"config set {key_path} deleted the next section's comments:\n"
        + "\n".join(lost)
    )


@pytest.mark.parametrize(
    ("doc", "key_path"),
    [
        pytest.param("a: 1\n\n# NOTE below\nb: 2\n", "a", id="top-level-first-key"),
        pytest.param("sec:\n  a: 1\n  # NOTE below\n  b: 2\n", "sec.a", id="nested-first-key"),
        pytest.param("sec:  # sec note\n  a:\n    x: 1\n\n  # NOTE below\n  b: 2\n", "sec.a", id="commented-parent-first-submap"),
    ],
)
def test_removing_first_key_of_non_empty_mapping_keeps_second_keys_comment(tmp_path, doc, key_path):
    """Removing the first key of a mapping (where ``prev`` is None) must move the lines that
    introduce the second key onto that second key instead of dropping them."""
    for writer in ("update", "save"):
        path = tmp_path / f"{writer}.yaml"
        path.write_text(doc, encoding="utf-8")
        if writer == "update":
            atomic_roundtrip_yaml_update(path, key_path, None)
        else:
            state = yaml.safe_load(doc)
            if "." in key_path:
                parent, child = key_path.split(".")
                state[parent].pop(child)
            else:
                state.pop(key_path)
            atomic_roundtrip_yaml_save(path, state)

        text = path.read_text(encoding="utf-8")
        assert "# NOTE below" in text, f"{writer} dropped second key's comment:\n{text}"
        assert text.index("# NOTE below") < text.index("b: 2"), text


@pytest.mark.parametrize(
    ("doc", "empty_write", "refill_write", "last_entry"),
    [
        pytest.param(
            "display:  # display note\n  skin: default\n\n# === Next Section ===\nagent:\n  max_turns: 60\n",
            lambda p: atomic_roundtrip_yaml_update(p, "display", {}),
            lambda p: atomic_roundtrip_yaml_update(p, "display.skin", "dark"),
            "skin: dark",
            id="update-empty-map-then-set-child",
        ),
        pytest.param(
            "display:  # display note\n  skin: default\n\n# === Next Section ===\nagent:\n  max_turns: 60\n",
            lambda p: atomic_roundtrip_yaml_save(p, {"display": {}, "agent": {"max_turns": 60}}),
            lambda p: atomic_roundtrip_yaml_save(p, {"display": {"skin": "dark"}, "agent": {"max_turns": 60}}),
            "skin: dark",
            id="save-empty-map-then-refill",
        ),
        pytest.param(
            "kanban:\n  dispatch_profiles:  # note\n    - mordecai\n\n  # interval note\n  dispatch_interval_seconds: 60\n",
            lambda p: atomic_roundtrip_yaml_update(p, "kanban.dispatch_profiles", []),
            lambda p: atomic_roundtrip_yaml_update(p, "kanban.dispatch_profiles", ["mordecai"]),
            "- mordecai",
            id="update-empty-list-then-refill",
        ),
        pytest.param(
            "sec:\n  x: 1\n\n# === Next Section ===\nb: 2\n",
            lambda p: None,
            lambda p: atomic_roundtrip_yaml_update(p, "sec", {"z": 2}),
            "z: 2",
            id="update-replace-map-with-plain-dict",
        ),
        pytest.param(
            "sec: null  # inline\n\n# === Next Section ===\nb: 2\n",
            lambda p: None,
            lambda p: atomic_roundtrip_yaml_update(p, "sec.sub.z", 2),
            "z: 2",
            id="update-scalar-to-nested-map",
        ),
    ],
)
def test_filling_or_replacing_collection_keeps_next_section_comment_after_it(
    tmp_path, doc, empty_write, refill_write, last_entry
):
    """When an empty/scalar key is filled or a collection is replaced, the next section's banner
    must follow the new collection's last entry rather than moving inside the collection above it."""
    path = tmp_path / "config.yaml"
    path.write_text(doc, encoding="utf-8")
    empty_write(path)
    refill_write(path)

    text = path.read_text(encoding="utf-8")
    banner = "# === Next Section ===" if "# === Next Section ===" in doc else "# interval note"
    assert banner in text, text
    assert text.index(last_entry) < text.index(banner), (
        f"next section comment moved inside collection above {last_entry!r}:\n{text}"
    )


@pytest.mark.parametrize(
    ("doc", "grow_write", "new_tail"),
    [
        pytest.param(
            "sec:\n  - a\n\n# === Next Section ===\nb: 2\n",
            lambda p: atomic_roundtrip_yaml_save(p, {"sec": ["a", "new_item"], "b": 2}),
            "- new_item",
            id="save-grow-list",
        ),
        pytest.param(
            "sec:\n  x: 1\n\n# === Next Section ===\nb: 2\n",
            lambda p: atomic_roundtrip_yaml_save(p, {"sec": {"x": 1, "z": 2}, "b": 2}),
            "z: 2",
            id="save-grow-map",
        ),
        pytest.param(
            "sec:\n  x: 1\n\n# === Next Section ===\nb: 2\n",
            lambda p: atomic_roundtrip_yaml_update(p, "sec.z", 2),
            "z: 2",
            id="update-grow-map",
        ),
        pytest.param(
            "sec:\n  x: 1\n\n# === Next Section ===\nb: 2\n",
            lambda p: atomic_roundtrip_yaml_update(p, "sec.sub.z", 2),
            "z: 2",
            id="update-grow-map-nested",
        ),
    ],
)
def test_growing_collection_moves_next_section_comment_to_new_tail(tmp_path, doc, grow_write, new_tail):
    """Appending to a list or adding a new key to a mapping must move the following section's
    banner to the new tail entry instead of leaving it between the old tail and the new entry."""
    path = tmp_path / "config.yaml"
    path.write_text(doc, encoding="utf-8")
    grow_write(path)

    text = path.read_text(encoding="utf-8")
    assert text.index(new_tail) < text.index("# === Next Section ==="), (
        f"next section comment was stranded before {new_tail!r}:\n{text}"
    )
