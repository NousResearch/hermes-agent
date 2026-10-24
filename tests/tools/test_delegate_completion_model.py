"""The async completion header names the model the children actually run on."""

from dataclasses import fields
from types import SimpleNamespace

from tools import async_delegation
from tools.delegate_tool_dispatch import _Batch, _dispatch_unit


def _unit(configured_model, *child_models):
    batch = _Batch(**{field.name: None for field in fields(_Batch)})
    batch.task_list = [{"goal": "probe"} for _ in child_models]
    batch.live_writers, batch.top_role = [], "leaf"
    batch.children = [(i, task, SimpleNamespace(model=model))
                      for i, (task, model) in enumerate(zip(batch.task_list, child_models))]
    batch.creds = {"model": configured_model}
    return batch


def test_completion_model_falls_back_to_inherited_child_model(monkeypatch):
    seen = []

    def dispatch(**kwargs):
        seen.append(kwargs["model"])
        return {"status": "dispatched", "delegation_id": "probe"}

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", dispatch)
    _dispatch_unit(_unit(None, "inherited/model"), None, None, {})
    _dispatch_unit(_unit("configured/model", "configured/model"), None, None, {})
    # Per-task models win over the global default; mixed units have no single model.
    _dispatch_unit(_unit("configured/model", "task/model"), None, None, {})
    _dispatch_unit(_unit(None, "first/model", "second/model"), None, None, {})
    assert seen == ["inherited/model", "configured/model", "task/model", None]
