"""mode='patch' without a body must name the missing argument and the
mode='replace' alternative (issue #125240)."""

from __future__ import annotations

import json


class TestPatchBodyRequiredHint:
    def test_error_names_patch_argument_and_replace_mode(self):
        from tools.file_tools import _handle_patch

        raw = _handle_patch({"mode": "patch"}, task_id="default")
        data = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else {"error": str(raw)}
        err = str(data.get("error", ""))
        assert "patch content required" in err
        assert "'patch' argument" in err
        assert "mode='replace'" in err
