"""Recovery hints must name the user-visible notebook, not its byte-transport copy."""
import json
from pathlib import Path
import shlex

import pytest

from tools.read_extract import (
    _needs_ocr_warning,
    _pdf_coverage_note,
    extract_document_bytes,
    extract_document_text,
)


@pytest.mark.parametrize('reader', ['bytes', 'text', 'read_file'])
def test_notebook_recovery_hint_opens_original_file(tmp_path, reader):
    path = tmp_path / "training team's notebook.ipynb"
    output = 'epoch progress\n' * 2000
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [{
        'cell_type': 'code', 'id': 'train', 'execution_count': 1, 'metadata': {},
        'source': ['print(log)'], 'outputs': [{
            'output_type': 'stream', 'name': 'stdout', 'text': output}]}]}
    path.write_text(json.dumps(notebook), encoding='utf-8')
    if reader == 'bytes':
        text = extract_document_bytes(path.read_bytes(), str(path))
    elif reader == 'text':
        text = extract_document_text(str(path))
    else:
        # Real public entry point, including configured local byte transport.
        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool(str(path), task_id='notebook-hint-recovery'))
        assert result.get('extracted_document'), result
        text = result['content']
    hint = next(line for line in text.splitlines() if 'full output: jq' in line)
    command = hint.split('full output: ', 1)[1].removesuffix(']')
    argv = shlex.split(command)
    assert argv == ['jq', '-r', '.cells[0].outputs', str(path)]
    recovered = json.loads(Path(argv[-1]).read_text(encoding='utf-8'))
    assert recovered['cells'][0]['outputs'][0]['text'] == output


def test_pdftoppm_recovery_hints_round_trip_apostrophe_path(monkeypatch):
    path = "/tmp/training team's notebook.pdf"
    monkeypatch.setattr('tools.read_extract._pdf_page_texts', lambda _p: ['x' * 500, '', '', ''])
    for note in (_needs_ocr_warning(path, [2]), _pdf_coverage_note('/tmp/copy.pdf', display_path=path)):
        command = note.split('`', 2)[1]
        argv = shlex.split(command)
        assert argv[-2] == path, argv


def test_writing_back_the_rendering_read_file_showed_cannot_replace_the_notebook(tmp_path):
    """read_file extracts a notebook to a text rendering and grants the write baseline (so an
    existing notebook can be overwritten). The model never saw the JSON, so writing that rendering
    back must be refused — a notebook write yields a parseable notebook or leaves the file alone —
    while a real nbformat overwrite still goes through without a re-read."""
    import re

    from tools.file_tools import read_file_tool, write_file_tool

    path = tmp_path / "analysis.ipynb"
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'cell_type': 'code', 'id': 'c1', 'execution_count': 1, 'metadata': {},
         'source': ['x = 41\n', 'print(x)'],
         'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': '41\n'}]}]}
    path.write_text(json.dumps(notebook), encoding='utf-8')

    read = json.loads(read_file_tool(str(path), task_id='nb-write-guard'))
    assert read.get('extracted_document'), read
    rendering = re.sub(r'(?m)^\s*\d+\|', '', read['content']).replace('x = 41', 'x = 42')
    refused = json.loads(write_file_tool(str(path), rendering, task_id='nb-write-guard'))

    assert refused.get('error') and not refused.get('files_modified'), refused
    assert json.loads(path.read_text(encoding='utf-8'))['cells'] == notebook['cells']

    notebook['cells'][0]['source'] = ['x = 42\n', 'print(x)']
    written = json.loads(write_file_tool(str(path), json.dumps(notebook), task_id='nb-write-guard'))
    assert not written.get('error'), written
    assert json.loads(path.read_text(encoding='utf-8'))['cells'][0]['source'][0] == 'x = 42\n'

    legacy = {'nbformat': 3, 'nbformat_minor': 0, 'metadata': {}, 'worksheets': [{'cells': [
        {'cell_type': 'code', 'language': 'python', 'input': 'x = 43', 'outputs': []}]}]}
    written = json.loads(write_file_tool(str(path), json.dumps(legacy), task_id='nb-write-guard'))
    assert not written.get('error'), written


_V4_CELLS = [{'cell_type': 'markdown', 'metadata': {}, 'source': '# kept'}]


@pytest.mark.parametrize('broken', [
    {'cells': []},
    {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'worksheets': [{'cells': _V4_CELLS}]},
    {'nbformat': 3, 'nbformat_minor': 0, 'metadata': {}, 'cells': _V4_CELLS},
    {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'cell_type': 'code', 'metadata': {}, 'source': 'x = 1'}]},
    {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'cell_type': 'markdown', 'source': '# no metadata'}]},
    {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'cell_type': 'text', 'metadata': {}, 'source': 'x'}]},
], ids=['cells-only', 'v4-worksheets', 'v3-cells', 'code-no-outputs', 'cell-no-metadata', 'bad-cell-type'])
def test_json_the_notebooks_version_schema_rejects_cannot_replace_the_notebook(tmp_path, broken):
    """A notebook write must be what nbformat accepts for the version it declares, not merely JSON
    with a ``cells`` key: anything missing the version's required top-level or cell fields is
    refused and the notebook on disk is untouched."""
    from tools.file_tools import read_file_tool, write_file_tool

    path = tmp_path / "analysis.ipynb"
    original = json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': _V4_CELLS})
    path.write_text(original, encoding='utf-8')
    assert json.loads(read_file_tool(str(path), task_id='nb-schema-guard')).get('extracted_document')

    refused = json.loads(write_file_tool(str(path), json.dumps(broken), task_id='nb-schema-guard'))

    assert refused.get('error') and not refused.get('files_modified'), refused
    assert path.read_text(encoding='utf-8') == original
