"""A third-party platform can declare file-only delivery using the published API."""

import hashlib
import json
import os
import subprocess
import sys
from textwrap import dedent

from hermes_cli.plugin_validate import validate_plugin_dir


def test_public_platform_policy_delivers_file_only_message(tmp_path):
    home = tmp_path / "home"
    plugin = home / "plugins" / "media-fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: media-fixture\nversion: 0.1.0\ndescription: fixture\nkind: platform\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - media-fixture\n", encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(dedent('''\
        import hashlib
        import json
        from pathlib import Path
        from gateway.platforms.send_targets import StandaloneSendPolicy

        async def send(pconfig, chat_id, message, *, thread_id=None, media_files=None):
            delivery = {"chat_id": chat_id, "message": message, "thread_id": thread_id,
                        "files": [{"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                   "is_voice": voice} for path, voice in media_files]}
            with Path(pconfig.extra["deliveries"]).open("a", encoding="utf-8") as output:
                output.write(json.dumps(delivery) + "\\n")
            return {"success": True, "message_id": "fixture-message"}

        def register(ctx):
            ctx.register_platform(
                name="media-fixture", label="Media fixture",
                adapter_factory=lambda config: None, check_fn=lambda: True,
                standalone_sender_fn=send,
                standalone_send=StandaloneSendPolicy(label="Media fixture", media=True),
            )
        '''), encoding="utf-8")
    report = validate_plugin_dir(plugin)
    assert report.ok, report.failures
    assert not [warning for warning in report.warnings if warning.startswith("private core imports")], report.warnings

    attachment = tmp_path / "document.txt"
    attachment.write_bytes(b"A standalone document without a caption.\n")
    deliveries = tmp_path / "deliveries.jsonl"
    script = dedent('''\
        import asyncio
        import json
        import sys
        from types import SimpleNamespace
        from gateway.config import Platform
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import discover_plugins
        from tools.send_message_tool import _send_to_platform

        discover_plugins()
        platform = Platform("media-fixture")
        entry = platform_registry.get(platform.value)
        pconfig = SimpleNamespace(extra={"deliveries": sys.argv[2]})
        policy, entry.standalone_send = entry.standalone_send, None
        unsupported = asyncio.run(_send_to_platform(
            platform, pconfig, "room-1", "", media_files=[(sys.argv[1], False)]))
        assert "error" in unsupported
        entry.standalone_send = policy
        result = asyncio.run(_send_to_platform(
            platform, pconfig, "room-1", "", thread_id="thread-1",
            media_files=[(sys.argv[1], False)]))
        print(json.dumps(result))
        ''')
    completed = subprocess.run(
        [sys.executable, "-c", script, str(attachment), str(deliveries)],
        env={**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": os.getcwd()},
        cwd=os.getcwd(), capture_output=True, text=True, check=True, timeout=60,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {"success": True, "message_id": "fixture-message"}
    assert [json.loads(line) for line in deliveries.read_text(encoding="utf-8").splitlines()] == [{
        "chat_id": "room-1", "message": "", "thread_id": "thread-1",
        "files": [{"sha256": hashlib.sha256(attachment.read_bytes()).hexdigest(), "is_voice": False}],
    }]
