"""``definitions`` -> ``$defs`` promotion in MCP tool schemas (``tools/mcp_tool_schema.py``).

A ``definitions``/``$defs`` value maps member NAMES to schemas: only the container key is
promoted, member names stay verbatim, and both spellings on one node merge into ``$defs``.
"""

from tools.mcp_tool_schema import _rewrite_local_refs


def test_definition_member_named_definitions_keeps_its_name():
    """A member *inside* a definitions map named ``definitions`` must survive.

    #56150 gated ``properties``/``patternProperties``, but a definition MAP's
    keys are member names too. The generic rewrite renamed a member literally
    named ``definitions`` to ``$defs``, while the ``$ref`` rewrite below turned
    ``#/definitions/definitions`` into ``#/$defs/definitions`` - so the promoted
    map no longer contained the name the reference pointed at, and the dangling
    $ref 400s the whole tool array.
    """
    out = _rewrite_local_refs({
        "type": "object",
        "definitions": {
            "definitions": {"type": "string"},
            "Other": {"type": "integer"},
        },
        "properties": {
            "a": {"$ref": "#/definitions/definitions"},
            "b": {"$ref": "#/definitions/Other"},
        },
    })

    # The meta-keyword is still promoted...
    assert "definitions" not in out
    # ...but the member NAME is verbatim, so the reference still resolves.
    assert set(out["$defs"]) == {"definitions", "Other"}
    assert out["properties"]["a"]["$ref"] == "#/$defs/definitions"
    assert out["properties"]["b"]["$ref"] == "#/$defs/Other"
    for prop in out["properties"].values():
        assert prop["$ref"].rsplit("/", 1)[1] in out["$defs"], "dangling $ref"


def test_definitions_and_defs_coexisting_merge_with_defs_winning():
    """A node carrying both spellings merges into one ``$defs``.

    Order-independent: ``$defs`` is the modern form, so it wins a name
    collision whichever key the producer emitted first.
    """
    for schema in (
        {"definitions": {"Legacy": {"type": "string"}, "Both": {"type": "string"}},
         "$defs": {"Modern": {"type": "integer"}, "Both": {"type": "integer"}}},
        {"$defs": {"Modern": {"type": "integer"}, "Both": {"type": "integer"}},
         "definitions": {"Legacy": {"type": "string"}, "Both": {"type": "string"}}},
    ):
        out = _rewrite_local_refs(dict(schema, type="object"))
        assert "definitions" not in out
        assert set(out["$defs"]) == {"Legacy", "Modern", "Both"}
        assert out["$defs"]["Both"] == {"type": "integer"}, "$defs must win the collision"


def test_nested_definitions_meta_keyword_still_promoted():
    """The gate must not stop promotion of a genuinely nested meta-keyword."""
    out = _rewrite_local_refs({
        "type": "object",
        "properties": {"p": {"definitions": {"X": {"type": "string"}},
                             "$ref": "#/definitions/X"}},
    })
    prop = out["properties"]["p"]
    assert prop["$defs"] == {"X": {"type": "string"}}
    assert prop["$ref"] == "#/$defs/X"
