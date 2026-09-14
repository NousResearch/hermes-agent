"""Focused renderer refusal and directionality contracts."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "scripts" / "gen_gateway_contracts.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_gateway_contracts_renderer", GEN)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _render_model(gen, model: type[BaseModel], *, inbound: bool = False) -> str:
    schemas = gen.ContractSchemas()
    ref = schemas.add(model, f"{model.__name__}.params", inbound=inbound)
    renderer = gen.Renderer(schemas.defs, schemas.inbound_defs)
    renderer.ensure(ref["$ref"].rsplit("/", 1)[1])
    return "\n".join(renderer.emitted.values())


def test_inbound_defaults_are_optional_but_outbound_defaults_are_present(gen):
    class Directional(BaseModel):
        required: str
        defaulted: str = "value"
        nullable: str | None = None

    inbound = _render_model(gen, Directional, inbound=True)
    outbound = _render_model(gen, Directional)

    assert "defaulted?: string\n" in inbound
    assert "nullable?: string | null\n" in inbound
    assert "defaulted: string\n" in outbound
    assert "nullable: string | null\n" in outbound


def test_shared_model_keeps_both_wire_directions(gen):
    class Shared(BaseModel):
        defaulted: str = "value"

    schemas = gen.ContractSchemas()
    outbound = schemas.add(Shared, "shared.result", inbound=False)
    inbound = schemas.add(Shared, "shared.params", inbound=True)
    renderer = gen.Renderer(schemas.defs, schemas.inbound_defs)
    renderer.ensure(outbound["$ref"].rsplit("/", 1)[1])
    renderer.ensure(inbound["$ref"].rsplit("/", 1)[1])
    rendered = "\n".join(renderer.emitted.values())

    assert "export interface SharedResult {\n  defaulted: string\n}" in rendered
    assert "export interface SharedParams {\n  defaulted?: string\n}" in rendered


def test_json_value_is_the_only_empty_schema_allowed(gen):
    class WithJsonValue(BaseModel):
        value: JsonValue

    assert "export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }\n" in (
        _render_model(gen, WithJsonValue)
    )

    class Empty(BaseModel):
        value: Any

    with pytest.raises(gen.RenderError, match=r"Empty\.value: empty schema"):
        _render_model(gen, Empty)


@pytest.mark.parametrize(
    ("field", "annotation", "kwargs", "match"),
    [
        ("minimum", int, {"ge": 0}, "minimum"),
        ("maximum", int, {"le": 1}, "maximum"),
        ("format", datetime, {}, "format"),
        ("minItems", list[str], {"min_length": 1}, "minItems"),
        ("maxItems", list[str], {"max_length": 1}, "maxItems"),
        ("pattern", str, {"pattern": "x"}, "pattern"),
        ("multipleOf", float, {"multiple_of": 0.5}, "multipleOf"),
    ],
)
def test_constraints_raise_render_error_with_model_field(gen, field, annotation, kwargs, match):
    model = type("Model", (BaseModel,), {"__annotations__": {field: annotation}, field: Field(**kwargs)})
    with pytest.raises(gen.RenderError, match=rf"Model\.{field}: keyword '{match}'"):
        _render_model(gen, model)


def test_closed_mapping_schema_renders_a_record(gen):
    class Mapping(BaseModel):
        values: dict[str, list[str]]

    assert "values: Record<string, string[]>\n" in _render_model(gen, Mapping)


def test_open_models_and_tuples_raise_render_error_with_model_field(gen):
    class Open(BaseModel):
        model_config = ConfigDict(extra="allow")
        value: str

    class BareMapping(BaseModel):
        value: dict

    class TupleModel(BaseModel):
        value: tuple[str, int]

    with pytest.raises(gen.RenderError, match=r"Open: additionalProperties: true"):
        _render_model(gen, Open)
    with pytest.raises(gen.RenderError, match=r"BareMapping\.value: additionalProperties: true"):
        _render_model(gen, BareMapping)
    with pytest.raises(gen.RenderError, match=r"TupleModel\.value: keyword '(maxItems|prefixItems)'"):
        _render_model(gen, TupleModel)


def test_unions_wrap_like_prettier_at_print_width(gen):
    # Members are 9 chars + quotes = 11 columns; n members join to 14n - 3 columns.
    short = Literal["a", "b"]
    eight = Literal[tuple(f"member_{index:02d}" for index in range(8))]  # 109 columns
    ten = Literal[tuple(f"member_{index:02d}" for index in range(10))]  # 137 columns

    schemas = gen.ContractSchemas()
    schemas.add(short, "wrap_inline.result", inbound=False)
    schemas.add(eight, "wrap_continuation.result", inbound=False)
    schemas.add(ten, "wrap_per_line.result", inbound=False)
    schemas.add(type("WrapField", (BaseModel,), {"__annotations__": {"kind": ten}}), "wrap_field.result", inbound=False)
    rendered = "\n".join(gen.Renderer(schemas.defs, schemas.inbound_defs).render_all().values())

    eight_union = " | ".join(f"'member_{index:02d}'" for index in range(8))
    ten_lines = "\n".join(f"  | 'member_{index:02d}'" for index in range(10))
    assert "export type WrapInlineResult = 'a' | 'b'\n" in rendered
    assert f"export type WrapContinuationResult =\n  {eight_union}\n" in rendered
    assert f"export type WrapPerLineResult =\n{ten_lines}\n" in rendered
    assert "export interface WrapFieldResult {\n  kind:\n" + ten_lines.replace("  |", "    |") + "\n}" in rendered
    assert all(len(line) <= 120 for line in rendered.splitlines())


def test_duplicate_class_names_raise_render_error(gen):
    left = type("Duplicate", (BaseModel,), {"__annotations__": {"left": str}})
    right = type("Duplicate", (BaseModel,), {"__annotations__": {"right": int}})
    schemas = gen.ContractSchemas()
    schemas.add(left, "left.params", inbound=True)

    with pytest.raises(gen.RenderError, match=r"right\.params: duplicate class name 'DuplicateInput'"):
        schemas.add(right, "right.params", inbound=True)
