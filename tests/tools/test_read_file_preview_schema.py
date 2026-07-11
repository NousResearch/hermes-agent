from tools.file_tools import READ_FILE_SCHEMA, read_file_tool


def test_read_file_preview_flag_is_optional_and_defaults_false():
    preview = READ_FILE_SCHEMA["parameters"]["properties"]["preview"]

    assert preview["type"] == "boolean"
    assert preview["default"] is False
    assert "preview" not in READ_FILE_SCHEMA["parameters"]["required"]


def test_read_file_preview_flag_requires_an_explicit_desktop_request():
    description = READ_FILE_SCHEMA["parameters"]["properties"]["preview"]["description"]

    assert description == (
        "Set true only when the user explicitly asks to open/show/display the text file "
        "in Desktop/Webapp Preview Workspace."
    )


def test_read_file_preview_flag_is_accepted_without_changing_the_read(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("alpha\nbeta\n", encoding="utf-8")

    ordinary = read_file_tool(str(sample), task_id="ordinary-read")
    requested_preview = read_file_tool(str(sample), task_id="preview-read", preview=True)

    assert requested_preview == ordinary
    assert "1|alpha" in requested_preview
    assert "2|beta" in requested_preview
