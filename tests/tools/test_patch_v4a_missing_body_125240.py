"""#125240: empty-body mode=patch must self-correct.

Schema stays required==["mode"] (anyOf/oneOf break sanitizers), so the
runtime rejection is the teaching moment: it must name 'patch', show the
V4A skeleton, and name the replace alternative.
"""
import json


class TestPatchV4aMissingBody125240:
    def test_empty_body_error_teaches_recovery(self):
        from tools.file_tools import patch_tool

        for kwargs in ({"mode": "patch"}, {"mode": "patch", "patch": ""}, {"mode": "patch", "patch": "   "}):
            err = json.loads(patch_tool(**kwargs)).get("error", "")
            assert "'patch'" in err, err
            assert "*** Begin Patch" in err, err
            assert "replace" in err.lower(), err
