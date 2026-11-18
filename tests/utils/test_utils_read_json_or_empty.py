"""``utils.read_json_or_empty``: the JSON object on disk, or ``{}`` for anything else."""

from utils import read_json_or_empty


def test_reads_an_object_and_treats_missing_damaged_or_non_object_files_as_empty(tmp_path):
    path = tmp_path / "store.json"
    assert read_json_or_empty(path) == {}
    path.write_text('{"task": "snapshot"}', encoding="utf-8")
    assert read_json_or_empty(path) == {"task": "snapshot"}
    path.write_text('["not a mapping"]', encoding="utf-8")
    assert read_json_or_empty(path) == {}
    path.write_text("{bad json", encoding="utf-8")
    assert read_json_or_empty(path) == {}
