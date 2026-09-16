"""Recovery hints must name the user-visible notebook, not its byte-transport copy."""
import json
from pathlib import Path
import shlex

import pytest

from tools.read_extract import extract_document_bytes, extract_document_text


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
