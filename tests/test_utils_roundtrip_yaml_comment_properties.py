"""Seeded properties for the comment-preserving config writers (#148524).

Comments in a config.yaml sit in many places ruamel tracks differently: after a key that opens a
block (``key:  # note``), on lines between a key and its entries, after list items, and as blank
lines. Every one of those layouts, under every kind of edit a settings write makes (emptying,
filling, replacing, shrinking, growing, deleting or nulling a value), must leave a file that reads
back as exactly the saved data, or refuse the write and leave the file as it was. Nothing else.

Generators are ``random.Random(seed)`` (hypothesis is not a dependency); each failure names its
seed, so ``-k`` on the seed reproduces it.
"""

from __future__ import annotations

import copy
import random

import pytest

import hermes_yaml as yaml

WORDS = ("alpha", "beta", "gamma", "delta", "theta")
# Leaves include empty flow collections: ``- {}`` opening a commented list is a no-op-save trigger.
LEAVES = (1, 2.5, True, False, None, "text", "two words", dict, list)


def _value(rng: random.Random, depth: int):
    roll = rng.random()
    if depth < 3 and roll < 0.3:
        return {f"{rng.choice(WORDS)}{i}": _value(rng, depth + 1) for i in range(rng.randint(0, 3))}
    if depth < 3 and roll < 0.55:
        return [_value(rng, depth + 2) for _ in range(rng.randint(0, 3))]
    leaf = rng.choice(LEAVES)
    return leaf() if leaf in (dict, list) else leaf  # a fresh object, or safe_dump writes anchors


def _render(rng: random.Random, value, indent: int = 0) -> list[str]:
    """Block YAML for *value* with comments and blank lines in every position ruamel tracks."""
    pad, lines = " " * indent, []

    def note() -> str:
        return f"# n{rng.randrange(150)}"

    def scalar(v) -> str:
        return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()

    if isinstance(value, dict):
        for key, child in value.items():
            if rng.random() < 0.2:
                lines.append(f"{pad}{note()}")
            if rng.random() < 0.1:
                lines.append("")
            if isinstance(child, (dict, list)) and child:
                lines.append(f"{pad}{key}:" + (f"   {note()}" if rng.random() < 0.4 else ""))
                if rng.random() < 0.25:
                    lines.append(f"{pad}  {note()}")
                lines += _render(rng, child, indent + 2)
            else:
                lines.append(f"{pad}{key}: {scalar(child)}" + (f"  {note()}" if rng.random() < 0.3 else ""))
    else:
        for child in value:
            eol = f"  {note()}" if rng.random() < 0.3 else ""
            if isinstance(child, (dict, list)) and child:
                lines.append(f"{pad}-{eol}")
                lines += _render(rng, child, indent + 2)
            else:
                lines.append(f"{pad}- {scalar(child)}{eol}")
    return lines


def _edit(rng: random.Random, state: dict) -> tuple:
    """Apply one random settings edit to *state* in place; returns the edited path."""
    paths, stack = [], [((), state)]
    while stack:
        prefix, node = stack.pop()
        for key, child in node.items():
            paths.append(prefix + (key,))
            if isinstance(child, dict):
                stack.append((prefix + (key,), child))
    path = rng.choice(paths)
    parent = state
    for key in path[:-1]:
        parent = parent[key]
    current = parent[path[-1]]
    edit = rng.choice(("empty", "scalar", "list", "map", "grow", "shrink", "delete", "null"))
    if edit == "empty":
        parent[path[-1]] = {} if isinstance(current, dict) else []
    elif edit == "scalar":
        parent[path[-1]] = rng.choice((0, "x", True, 3.5))
    elif edit == "list":
        parent[path[-1]] = [rng.choice(WORDS) for _ in range(rng.randint(0, 2))]
    elif edit == "map":
        parent[path[-1]] = {rng.choice(WORDS): 1}
    elif edit == "grow" and isinstance(current, list):
        current.append(rng.choice(WORDS))
    elif edit in ("shrink", "grow") and isinstance(current, (list, dict)) and current:
        del current[rng.randrange(len(current)) if isinstance(current, list) else rng.choice(list(current))]
    elif edit == "delete":
        del parent[path[-1]]
    else:
        parent[path[-1]] = None
    return path


def _case(seed: int):
    rng = random.Random(seed)
    data = {f"{rng.choice(WORDS)}{i}": _value(rng, 0) for i in range(rng.randint(1, 4))}
    text = "\n".join(_render(rng, data)) + "\n"
    if yaml.safe_load(text) != data:  # a generator corner (e.g. an empty block) — not a writer case
        pytest.skip("generator produced a document that does not encode its data")
    return rng, data, text


def _assert_saved_or_refused(config_path, original: str, intended: dict, write, seed: int) -> None:
    try:
        write()
    except RuntimeError as exc:
        assert "this change was not saved" in str(exc), f"seed {seed}: {exc}"
        assert config_path.read_text(encoding="utf-8") == original, f"seed {seed}: refused but changed"
        return
    written = config_path.read_text(encoding="utf-8")
    try:
        reread = yaml.safe_load(written)
    except yaml.YAMLError as exc:
        pytest.fail(f"seed {seed}: config.yaml no longer parses ({exc})\n--- before\n{original}--- after\n{written}")
    assert reread == intended, f"seed {seed}\n--- before\n{original}--- after\n{written}"


@pytest.mark.parametrize("seed", range(150))
def test_full_state_save_never_corrupts_a_commented_config(seed, tmp_path):
    from utils import atomic_roundtrip_yaml_save

    rng, data, original = _case(seed)
    intended = copy.deepcopy(data)
    for _ in range(rng.randint(1, 3)):
        if intended:
            _edit(rng, intended)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(original, encoding="utf-8")

    _assert_saved_or_refused(
        config_path, original, intended, lambda: atomic_roundtrip_yaml_save(config_path, intended), seed)


@pytest.mark.parametrize("seed", range(150))
def test_single_key_update_never_corrupts_a_commented_config(seed, tmp_path):
    from utils import atomic_roundtrip_yaml_update

    rng, data, original = _case(seed)
    edited = copy.deepcopy(data)
    top = _edit(rng, edited)[0]
    value = edited.get(top)
    intended = {k: v for k, v in data.items() if k != top}
    if value is not None:  # the update writer removes a key set to None
        intended[top] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(original, encoding="utf-8")

    _assert_saved_or_refused(
        config_path, original, intended, lambda: atomic_roundtrip_yaml_update(config_path, top, value), seed)


def test_noop_save_preserves_every_commented_config(tmp_path):
    """Re-saving unchanged settings (every unrelated write does) must change nothing. One test over
    many seeds: the emitter defect this guards against shows up in about 1 layout in 120."""
    from utils import atomic_roundtrip_yaml_save

    config_path, changed = tmp_path / "config.yaml", []
    for seed in range(300):  # main corrupts seeds 288 and 298 here
        rng = random.Random(seed)
        data = {f"{rng.choice(WORDS)}{i}": _value(rng, 0) for i in range(rng.randint(1, 4))}
        original = "\n".join(_render(rng, data)) + "\n"
        if yaml.safe_load(original) != data:
            continue
        config_path.write_text(original, encoding="utf-8")
        atomic_roundtrip_yaml_save(config_path, data)
        try:
            if yaml.safe_load(config_path.read_text(encoding="utf-8")) != data:
                changed.append(seed)
        except yaml.YAMLError:
            changed.append(seed)

    assert not changed, f"a no-op save changed the settings of seeds {changed}"
