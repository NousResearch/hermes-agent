"""Shipped Python bytecode never survives into a hub skill bundle."""

from tools.skills_hub_models import SkillBundle


def test_shipped_python_bytecode_never_reaches_disk():
    # An unchecked-hash .pyc is imported instead of the scanned .py beside it, so a bundle
    # from ANY source must lose its bytecode before quarantine, scan and content hashing.
    bundle = SkillBundle(
        name="csv-helper",
        files={
            "SKILL.md": "---\nname: csv-helper\n---\n",
            "scripts/helper.py": "def run():\n    return 'ok'\n",
            "scripts/__pycache__/helper.cpython-314.pyc": b"\x00payload",
            "scripts\\legacy.pyc": b"\x00payload",
        },
        source="url",
        identifier="https://example.com/csv-helper/SKILL.md",
        trust_level="community",
    )

    assert sorted(bundle.files) == ["SKILL.md", "scripts/helper.py"]
