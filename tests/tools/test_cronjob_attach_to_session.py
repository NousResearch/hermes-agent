"""Regression tests: the registered cronjob handler must forward every
schema-declared parameter to the cronjob() implementation.

The tool schema declared ``attach_to_session`` and cronjob() accepted it, but
the ``handler`` lambda in registry.register() never passed it through — so
``cronjob(action='create'/'update', attach_to_session=True)`` from an agent
silently dropped the flag: the job was created without it and an update-only
call returned "No updates provided."
"""
import inspect
import json
from unittest.mock import patch

import tools.cronjob_tools  # noqa: F401 — triggers registry.register()
from tools.registry import registry


def _dispatch(args: dict) -> dict:
    entry = registry.get_entry("cronjob")
    assert entry is not None, "cronjob tool not registered"
    return json.loads(entry.handler(args))


class TestHandlerForwardsAttachToSession:
    def test_create_forwards_attach_to_session(self):
        captured = {}

        def fake_create_job(**kwargs):
            captured.update(kwargs)
            return {"id": "job-attach-1", "name": "t", "schedule": {"kind": "cron", "expr": "0 9 * * *"},
                    "schedule_display": "0 9 * * *", "repeat": None, "deliver": "discord",
                    "next_run_at": None, "attach_to_session": True}

        with patch("tools.cronjob_tools.create_job", side_effect=fake_create_job):
            out = _dispatch({
                "action": "create",
                "schedule": "0 9 * * *",
                "prompt": "daily briefing",
                "deliver": "discord",
                "attach_to_session": True,
            })

        assert out.get("success", True) is not False, out
        assert captured.get("attach_to_session") is True, (
            "handler dropped attach_to_session on create: %r" % (captured,)
        )

    def test_update_only_attach_to_session_is_not_empty(self):
        """An update that ONLY sets attach_to_session must not be rejected
        as 'No updates provided.'"""
        job = {"id": "job-attach-2", "name": "t", "prompt": "x",
               "schedule": {"kind": "cron", "expr": "0 9 * * *"},
               "schedule_display": "0 9 * * *"}
        captured = {}

        def fake_update_job(job_id, updates):
            captured["job_id"] = job_id
            captured["updates"] = updates
            return {**job, **updates}

        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(job)), \
             patch("tools.cronjob_tools.update_job", side_effect=fake_update_job):
            out = _dispatch({
                "action": "update",
                "job_id": "job-attach-2",
                "attach_to_session": True,
            })

        assert out.get("success", True) is not False, out
        assert captured.get("updates", {}).get("attach_to_session") is True, (
            "handler dropped attach_to_session on update: %r" % (captured,)
        )


class TestHandlerCoversSchemaParams:
    def test_every_schema_param_reaches_cronjob_signature(self):
        """Guard against the next silently-dropped parameter: every property
        declared in the tool schema must be an accepted kwarg of cronjob()."""
        entry = registry.get_entry("cronjob")
        assert entry is not None
        schema_props = set(entry.schema["parameters"]["properties"].keys())
        sig_params = set(inspect.signature(tools.cronjob_tools.cronjob).parameters.keys())
        missing = schema_props - sig_params
        assert not missing, f"schema params not accepted by cronjob(): {missing}"
