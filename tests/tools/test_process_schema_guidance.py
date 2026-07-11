from tools.process_registry import PROCESS_SCHEMA


def test_process_schema_forbids_invented_or_synchronous_session_ids():
    description = PROCESS_SCHEMA["description"]
    session_description = PROCESS_SCHEMA["parameters"]["properties"]["session_id"][
        "description"
    ]

    assert "terminal(background=true)" in description
    assert "finished synchronously" in description
    assert "Never invent" in description
    assert "not_found" in description
    assert "terminal(background=true)" in session_description
