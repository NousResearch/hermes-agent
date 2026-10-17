"""Azure schema regression: exact RunPod patterns, no lost constraints or auth."""
import copy
import json
from pathlib import Path
import re
import unicodedata

import pytest

from tools.azure_schema import (
    AzureToolSchemaError, prepare_azure_request, sanitize_azure_tool_schemas,
)

PATTERN = r"[^\p{Z}\t-\r\x85]".replace(r"\x85", "\x85")


def tool(pattern=PATTERN, *, flat=False):
    fn = {"name": "mcp_runpod_create_global_volume", "parameters": {
        "type": "object", "properties": {"body": {"$ref": "#/$defs/Body"}},
        "required": ["body"], "$defs": {"Body": {
            "type": "object", "required": ["name", "size"], "additionalProperties": False,
            "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 191, "pattern": pattern},
                           "size": {"type": "integer", "minimum": 10, "maximum": 4096},
                           "kind": {"enum": ["STANDARD", "HIGH_PERFORMANCE"]}},
        }}}}
    return {"type": "function", **fn} if flat else {"type": "function", "function": fn}


def get_pattern(t):
    return t.get("function", t)["parameters"]["$defs"]["Body"]["properties"]["name"]["pattern"]


@pytest.mark.parametrize("flat", [False, True])
def test_runpod_unicode_z_preserves_every_non_pattern_constraint(flat):
    original = [tool(flat=flat)]
    before = copy.deepcopy(original)
    result = sanitize_azure_tool_schemas(original)
    assert original == before
    replacement = get_pattern(result[0])
    assert r"\p{Z}" not in replacement
    before[0].get("function", before[0])["parameters"]["$defs"]["Body"]["properties"]["name"]["pattern"] = replacement
    assert result == before
    assert sanitize_azure_tool_schemas(result) == result


def test_all_unicode_codepoints_preserve_runpod_non_whitespace_semantics():
    compiled = re.compile(get_pattern(sanitize_azure_tool_schemas([tool()])[0]))
    for cp in range(0x110000):
        char = chr(cp)
        expected = not (unicodedata.category(char).startswith("Z") or '\t' <= char <= '\r' or char == '\x85')
        assert bool(compiled.search(char)) == expected, hex(cp)


@pytest.mark.parametrize("name,valid", [("", False), (" \t\n\r\x85\u00a0\u3000", False),
    ("dataset", True), (" leading", True), ("trailing ", True), ("项目", True), ("💾", True), ("\u200b", True)])
def test_names_remain_search_not_fullmatch(name, valid):
    assert bool(re.search(get_pattern(sanitize_azure_tool_schemas([tool()])[0]), name)) is valid


@pytest.mark.parametrize("pattern,yes,no", [
    (r"^\p{Z}+$", " \u3000", "\t"), (r"^\P{Z}+$", "\tname", "\u00a0"),
    (r"^[\p{Z}a]+$", " a\u2009", "b"),
    (r"^\\p\{Z\}$", r"\p{Z}", " "),
    (r"^\[\p{Z}\]$", "[ ]", "[a]"),
])
def test_class_context_escaped_slashes_and_brackets(pattern, yes, no):
    compiled = re.compile(get_pattern(sanitize_azure_tool_schemas([tool(pattern)])[0]))
    assert compiled.search(yes)
    assert not compiled.search(no)


@pytest.mark.parametrize("pattern", [r"\p{Unknown}", r"[\P{Z}]", r"[", 42, r"[a-\p{Z}]", r"[\p{Z}-a]"])
def test_unknown_or_ambiguous_patterns_fail_closed_without_exposing_value(pattern):
    with pytest.raises(AzureToolSchemaError) as e:
        sanitize_azure_tool_schemas([tool(pattern)])
    assert str(pattern) not in str(e.value)


def test_metadata_and_property_names_are_data_not_schema():
    t = tool()
    params = t['function']['parameters']
    params['properties']['pattern'] = {'type': 'string'}
    params['default'] = {'pattern': r'\p{Unknown}'}
    params['examples'] = [{'type': 'string', 'pattern': '['}]
    params['description'] = r'\p{Unknown}'
    result = sanitize_azure_tool_schemas([t])[0]['function']['parameters']
    for key in ('default', 'examples', 'description', 'properties'):
        assert result[key] == params[key]


@pytest.mark.parametrize('keyword', ['items', 'additionalProperties', 'unevaluatedProperties', 'propertyNames', 'not', 'if', 'then', 'else', 'contains'])
def test_nested_schema_keywords_without_explicit_type(keyword):
    t = {'type':'function', 'name':'nested', 'parameters': {keyword: {'pattern': r'\p{Z}'}}}
    result = sanitize_azure_tool_schemas([t])
    assert r'\p{Z}' not in result[0]['parameters'][keyword]['pattern']


@pytest.mark.parametrize('keyword', ['anyOf', 'allOf', 'oneOf', 'prefixItems', 'items'])
def test_schema_arrays(keyword):
    t = {'type':'function', 'name':'nested', 'parameters': {keyword: [{'pattern': r'\p{Z}'}]}}
    assert r'\p{Z}' not in sanitize_azure_tool_schemas([t])[0]['parameters'][keyword][0]['pattern']


def test_pattern_property_keys_and_dependent_schemas():
    params = {'patternProperties': {r'\p{Z}': {'pattern': r'\p{Z}'}}, 'dependentSchemas': {'pattern': {'pattern': r'\p{Z}'}}}
    t = {'type':'function', 'name':'nested', 'parameters': params}
    changed = sanitize_azure_tool_schemas([t])[0]['parameters']
    key = next(iter(changed['patternProperties']))
    assert key != r'\p{Z}'
    assert changed['dependentSchemas']['pattern']['pattern'] == key


@pytest.mark.parametrize('provider,url', [('azure-foundry', ''), ('custom', 'https://demo.openai.azure.com/openai/v1/'), ('custom', 'https://demo.services.ai.azure.com/openai/v1/')])
def test_request_gate_preserves_model_credentials_and_non_tools(provider, url):
    sentinel = object()
    kwargs = {'model': 'gpt-6-astra', 'tools': [tool()], 'messages': [{'role':'user','content':'unchanged'}],
              'extra_headers': {'Authorization': sentinel}, 'extra_body': {'tools':[tool(flat=True)], 'other': sentinel}}
    before = get_pattern(kwargs['tools'][0])
    result = prepare_azure_request(kwargs, provider=provider, base_url=url)
    assert get_pattern(result['tools'][0]) != before
    assert get_pattern(result['extra_body']['tools'][0]) != before
    assert result['model'] == kwargs['model']
    assert result['messages'] is kwargs['messages']
    assert result['extra_headers'] is kwargs['extra_headers']
    assert result['extra_body']['other'] is sentinel
    assert get_pattern(kwargs['tools'][0]) == before


def test_non_azure_provider_is_untouched():
    kwargs = {'tools':[tool(r'\p{Unknown}')]}
    assert prepare_azure_request(kwargs, provider='openrouter', base_url='https://openai.azure.com.attacker.test/v1') is kwargs


def test_builtin_non_function_tools_are_preserved():
    t = {'type':'web_search_preview'}
    assert sanitize_azure_tool_schemas([t]) == [t]


def test_captured_runpod_regression_schemas():
    fixtures = json.loads((Path(__file__).parent / 'fixtures/runpod_azure_schemas.json').read_text())
    for item in fixtures:
        t = {'type':'function','function':{'name':item['name'], 'parameters':item['inputSchema']}}
        result = sanitize_azure_tool_schemas([t])
        # The captured create/update definitions retain their $refs and constraints.
        def erase_patterns(node):
            if isinstance(node, dict): return {k:erase_patterns(v) for k,v in node.items() if k != 'pattern'}
            if isinstance(node, list): return [erase_patterns(v) for v in node]
            return node
        assert erase_patterns(result) == erase_patterns([t])
        assert sanitize_azure_tool_schemas(result) == result


@pytest.mark.parametrize('pattern', [r'^[^]\p{Z}$', r'^[]\p{Z}$', r'[^]'])
def test_ambiguous_ecmascript_empty_classes_fail_closed(pattern):
    with pytest.raises(AzureToolSchemaError):
        sanitize_azure_tool_schemas([tool(pattern)])


@pytest.mark.parametrize('extra', [False, True])
def test_tuple_tool_collection_is_preflighted(extra):
    value = (tool(),)
    kwargs = {'extra_body':{'tools':value}} if extra else {'tools':value}
    result = prepare_azure_request(kwargs,provider='azure-foundry')
    sent = result['extra_body']['tools'] if extra else result['tools']
    assert get_pattern(sent[0]) != PATTERN
    assert get_pattern(value[0]) == PATTERN
    bad = {'tools':(tool(r'\p{Unknown}'),)}
    with pytest.raises(AzureToolSchemaError):
        prepare_azure_request({'extra_body':bad} if extra else bad,provider='azure-foundry')


@pytest.mark.parametrize('bad', ['not-a-list', {}, 42])
def test_malformed_tool_collection_fails_closed(bad):
    with pytest.raises(AzureToolSchemaError):
        prepare_azure_request({'tools':bad},provider='azure-foundry')


def test_sdk_omission_sentinels_and_null_are_preserved():
    from openai import NOT_GIVEN, omit
    for value in (None, NOT_GIVEN, omit):
        assert prepare_azure_request({'tools':value},provider='azure-foundry')['tools'] is value
