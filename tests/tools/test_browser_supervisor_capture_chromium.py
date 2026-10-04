"""Disposable Chromium smoke through the public capture API only."""
import shutil
import subprocess
import time
import pytest
from tools.browser_supervisor import SUPERVISOR_REGISTRY

@pytest.mark.skipif(not shutil.which('chromium'), reason='Disposable Chromium unavailable')
def test_public_capture_real_chromium(tmp_path):
    profile = tmp_path/'chromium'
    proc = subprocess.Popen([shutil.which('chromium'), '--headless=new', '--no-sandbox',
        '--disable-gpu', '--no-first-run', '--remote-debugging-port=0',
        '--remote-debugging-address=127.0.0.1', '--user-data-dir='+str(profile),
        'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    task = 'public-cdp-disposable-fixture'
    try:
        port_file = profile/'DevToolsActivePort'
        deadline = time.monotonic()+10
        while not port_file.exists():
            assert proc.poll() is None and time.monotonic()<deadline
            time.sleep(.05)
        port, path = port_file.read_text().splitlines()[:2]
        sup = SUPERVISOR_REGISTRY.get_or_start(task, f'ws://127.0.0.1:{port}{path}')
        assert sup.snapshot().active
        capture = SUPERVISOR_REGISTRY.capture(task, timeout=5)
        value = capture.call('Runtime.evaluate', {'expression':'"fixture transport"', 'returnByValue':True},
                             session_id=capture.page_session_id, timeout=5)
        assert value['result']['result']['value']=='fixture transport'
        child = capture.call('Target.createTarget', {'url':'about:blank'}, session_id=None, timeout=5)
        target = child['result']['targetId']
        attached = capture.call('Target.attachToTarget', {'targetId':target,'flatten':True},
                                session_id=None,timeout=5)['result']['sessionId']
        result = capture.call('Runtime.evaluate', {'expression':'document.URL','returnByValue':True},
                              session_id=attached, timeout=5)
        assert result['result']['result']['value']=='about:blank'
        capture.cleanup_call('Target.detachFromTarget', {'sessionId':attached}, session_id=None,timeout=5)
        assert capture.call('Target.closeTarget', {'targetId':target},session_id=None,timeout=5)['result']['success']
        assert capture.valid
    finally:
        SUPERVISOR_REGISTRY.stop(task)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
