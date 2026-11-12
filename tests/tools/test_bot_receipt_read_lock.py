"""Receipt polling joins the writer's short lock even when replacement rewrites in place."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import threading

import pytest

from tools.bot_live_delivery import _locked, _write, read_delivery_result


def test_reader_never_observes_a_contended_partial_rewrite(tmp_path):
    key = 'a' * 32
    record = dict(delivery_id=key, admission_id='owned', status='queued', reply='')
    with _locked(tmp_path) as root:
        _write(root / f'{key}.json', record)
    writing, finish = threading.Event(), threading.Event()
    completed = dict(record, status='settled', reply='complete owner reply')
    encoded = json.dumps(completed)
    def rewrite():
        with _locked(tmp_path) as root, (root / f'{key}.json').open('w', encoding='utf-8') as stream:
            middle = len(encoded) // 2
            stream.write(encoded[:middle])
            stream.flush()
            writing.set()
            assert finish.wait(10)
            stream.write(encoded[middle:])
    with ThreadPoolExecutor(2) as pool:
        writer = pool.submit(rewrite)
        assert writing.wait(5)
        reader = pool.submit(read_delivery_result, tmp_path, key)
        try:
            with pytest.raises(TimeoutError):
                reader.result(timeout=2)
        finally:
            finish.set()
            writer.result(timeout=5)
        assert reader.result(timeout=5) == completed
