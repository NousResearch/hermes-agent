"""Own one real headless Chromium and its DevTools connection for native tests."""
from pathlib import Path
import socket
import subprocess
from urllib.parse import quote

import pm
import psutil
import requests

from hermes_cli._subprocess_compat import windows_hide_flags
from tests.waiting import wait_until


class OwnedChrome:
    def __init__(self, directory: Path):
        installed = pm.installed_package('chromium')
        assert installed is not None and installed.binary is not None
        executable = Path(installed.binary)
        assert executable.is_file()
        directory.mkdir(parents=True)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]
        self.root = f'http://127.0.0.1:{self.port}'
        self.http = requests.Session()
        self.http.trust_env = False
        self.log = (directory / 'chrome.log').open('wb')
        self.process = subprocess.Popen(
            [str(executable), '--headless=new', '--no-first-run', '--no-default-browser-check',
             '--disable-background-networking', '--remote-debugging-address=127.0.0.1',
             f'--remote-debugging-port={self.port}', f'--user-data-dir={directory / "user-data"}', 'about:blank'],
            stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log, creationflags=windows_hide_flags(),
        )
        self.owner = psutil.Process(self.process.pid)
        self.owner_started = self.owner.create_time()
        try:
            self.websocket = wait_until(self._ready, timeout=30, what='the owned Chromium DevTools endpoint')
        except BaseException:
            self.stop()
            raise

    def _ready(self):
        assert self.process.poll() is None, 'The owned Chromium exited during startup'
        try:
            response = self.http.get(self.root + '/json/version', timeout=1)
        except requests.ConnectionError:
            return None
        response.raise_for_status()
        return response.json()['webSocketDebuggerUrl']

    def targets(self):
        response = self.http.get(self.root + '/json/list', timeout=5)
        response.raise_for_status()
        return {row['id']: row for row in response.json() if row['type'] == 'page'}

    def new_page(self, url):
        response = self.http.put(self.root + '/json/new?' + quote(url, safe=''), timeout=10)
        response.raise_for_status()
        target = response.json()['id']
        # Chrome creates the target before its first navigation commits.
        wait_until(lambda: self.targets().get(target, {}).get('url') == url,
                   timeout=10, what='the owned page navigation to commit')
        return target

    def stop(self):
        if self.process.poll() is None:
            assert self.owner.create_time() == self.owner_started
            children = self.owner.children(recursive=True)
            self.owner.terminate()
            self.owner.wait(timeout=10)
            for child in children:
                if child.is_running():
                    child.terminate()
            _, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                child.kill()
            _, alive = psutil.wait_procs(alive, timeout=10)
            assert not alive
        self.http.close()
        self.log.close()
